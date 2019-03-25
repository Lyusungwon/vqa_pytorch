import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils import *
from configuration import get_config
from recorder import Recorder


args, model, train_loader, test_loader = get_config()
device = args.device
torch.manual_seed(args.seed)

if not args.te_bert:
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
else:
    # not_bert = [param for name, param in model.named_parameters() if 'text_encoder' not in name]
    # optimizer = optim.Adam(not_bert,
    #                        lr=args.lr,
    #                        weight_decay=args.weight_decay)
    # from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
    # # num_train_optimization_steps = int(
    # #     len(train_loader) / args.batch_size) * args.epochs
    # yes_bert = [(name, param) for name, param in model.named_parameters() if 'text_encoder' in name]
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in yes_bert if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in yes_bert if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    #     ]
    # bert_optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=args.lr_bert,
    #                      warmup=-1,
    #                      t_total=-1)
    from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
    yes_bert = [(name, param) for name, param in model.named_parameters() if 'text_encoder' in name]
    no_bert = [param for name, param in model.named_parameters() if 'text_encoder' not in name]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in yes_bert if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in yes_bert if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': no_bert, 'lr': args.lr, 'weight_decay': args.weight_decay}
        ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.lr_bert,
                         warmup=-1,
                         t_total=-1)

start_epoch = 0
batch_record_idx = 0
if args.load_model:
    model, optimizer, start_epoch, batch_record_idx = load_checkpoint(model, optimizer, args.log, device)

# if args.lr_reduce:
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                      mode='min',
#                                                      factor=0.5,
#                                                      threshold=1e-8 if args.multi_label else 1e-6)
# elif args.lr_increase:
#     scheduler = optim.lr_scheduler.StepLR(optimizer,
#                                           step_size=20,
#                                           gamma=2)

criterion = F.binary_cross_entropy_with_logits if args.multi_label else F.cross_entropy
model = nn.DataParallel(model, device_ids=args.gpu_num) if args.multi_gpu else model
model = model.to(device)


def epoch(epoch_idx, is_train):
    model.train() if is_train else model.eval()
    loader = train_loader if is_train else test_loader
    recorder.epoch_start(epoch_idx, is_train, loader)
    for batch_idx, (image, question_set, answer, types) in enumerate(loader):
        batch_size = image.size()[0]
        optimizer.zero_grad()
        # if args.te_bert:
        #     bert_optimizer.zero_grad()
        image = image.to(device)
        question = question_set[0].to(device)
        question_length = question_set[1].to(device)
        answer = answer.to(device)
        output = model(image, question, question_length)
        loss = criterion(output, answer)
        if is_train:
            loss.backward()
            if args.gradient_clipping:
                nn.utils.clip_grad_value_(model.parameters(), args.gradient_clipping)
            optimizer.step()
            # if args.te_bert:
            #     bert_optimizer.step()
        recorder.batch_end(loss.item(), output.cpu().detach(), answer.cpu(), types.cpu())
        if is_train and (batch_idx % args.log_interval == 0):
            recorder.log_batch(batch_idx, batch_size)
    recorder.log_epoch()
    if not is_train:
        recorder.log_text(question.cpu(), output.cpu(), answer.cpu(), types.cpu())
        if not args.cv_pretrained:
            recorder.log_image(image.cpu())


if __name__ == '__main__':
    writer = SummaryWriter(args.log)
    recorder = Recorder(writer, args, batch_record_idx)
    for epoch_idx in range(start_epoch + 1, args.epochs + 1):
        epoch(epoch_idx, is_train=True)
        epoch(epoch_idx, is_train=False)
        module = model.module if args.multi_gpu else model
        save_checkpoint(epoch_idx, module, optimizer, args, recorder.batch_record_idx)
        # recorder.log_lr(optimizer.param_groups[0]['lr'])
        # if args.lr_reduce:
        #     scheduler.step(recorder.get_epoch_loss())
        # elif args.lr_increase:
        #     if optimizer.param_groups[0]['lr'] < args.lr_max:
        #         scheduler.step()
        # if epoch_idx % 20 == 0:
        #     recorder.log_embedding(module.text_encoder.embedding.weight.data)
    recorder.finish()
    writer.close()
