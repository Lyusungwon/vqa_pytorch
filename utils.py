import os
import torch
import torch.nn as nn
import time
import h5py
from text_preprocessor import tokenize_rm


def timefn(fn):
    def wrap(*args):
        t1 = time.time()
        result = fn(*args)
        t2 = time.time()
        print("@timefn:{} took {} seconds".format(fn.__name__, t2 - t1))
        return result
    return wrap


def is_file_exist(file):
    if os.path.isfile(file):
        print(f"Data {file} exist")
        return True
    else:
        print(f"Data {file} does not exist")
        return False


def save_checkpoint(epoch_idx, model, optimizer, args, batch_record_idx):
    log = args.log
    checkpoint = dict()
    checkpoint['model'] = model
    checkpoint['model_parameters'] = model.state_dict()
    checkpoint['optimizer_parameters'] = optimizer.state_dict()
    checkpoint['args'] = args
    checkpoint['epoch'] = epoch_idx
    checkpoint['batch_idx'] = batch_record_idx
    save_file = os.path.join(log, 'checkpoint.pt')
    torch.save(checkpoint, save_file)
    print('Model saved in {}'.format(save_file))
    return True


def load_checkpoint(model, optimizer, log, device):
    load_file = os.path.join(log, 'checkpoint.pt')
    checkpoint = torch.load(load_file)
    model.load_state_dict(checkpoint['model_parameters'])
    optimizer.load_state_dict(checkpoint['optimizer_parameters'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    epoch_idx = checkpoint['epoch']
    batch_record_idx = checkpoint['batch_idx']
    print('Model loaded from {}.'.format(load_file))
    return model, optimizer, epoch_idx, batch_record_idx


def load_pretrained_embedding(idx2word, embedding_dim):
    import torchtext
    pretrained = torchtext.vocab.GloVe(name='6B', dim=embedding_dim)
    embedding = torch.Tensor(len(idx2word), embedding_dim)
    missing = 0
    for idx, word in idx2word.items():
        if word != '<pad>':
            words = tokenize_rm(word)
            if words:
                word = words[0]
                embedding[idx, :] = pretrained[word].data
                if sum(embedding[idx, :]) == 0:
                    missing += 1
    # embedding = F.normalize(embedding, -1)
    print(f"Loaded pretrained embedding({(len(idx2word) - missing)/len(idx2word)}).")
    return embedding

def load_bert(num_labels=None):
    from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
    from pytorch_pretrained_bert.modeling import BertModel
    model = BertModel.from_pretrained('bert-base-uncased')
    # from pytorch_pretrained_bert.modeling import BertForSequenceClassification
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
    #           cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
    #           num_labels=num_labels)
    print("Loaded pretrained Bert.")
    return model


def load_pretrained_conv():
    import torchvision.models as models
    model = models.resnet101(pretrained=True)
    feature_extractor = list(model.children())[:-2]
    feature_extractor.append(nn.AvgPool2d(2, 2, 0))
    feature_extractor = nn.Sequential(*feature_extractor)
    feature_extractor.eval()
    print("Loaded pretrained feature extraction model.")
    return feature_extractor


def load_dict(args):
    with h5py.File(os.path.join(args.data_directory, args.dataset, f'qa_sets_{args.dataset}_train.h5'), 'r', swmr=True) as f:
        if args.dataset == 'vqa2':
            # args.word_to_idx = data_dict['word_to_idx']
            args.idx_to_word = f['question'][args.q_tokenizer]['dict'][:]
            # args.answer_word_to_idx = data_dict['answer_word_to_idx']
            args.idx_to_answer_word = f['answer']['uni-label' if not args.multi_label else 'multi-label'][args.a_tokenizer]['dict'][:]
            # args.question_type_to_idx = data_dict['question_type_to_idx']
            args.idx_to_question_type = f['question_type']['dict'][:]
        else:
            args.idx_to_word = f['question']['dict'][:]
            args.idx_to_answer_word = f['answer']['dict'][:]
            args.idx_to_question_type = f['question_type']['dict'][:]
    return args


def to_onehot(a, a_size, mask):
    onehot = torch.zeros(len(a), a_size)
    divide = 3.0 if len(a) > 1 else 1.0
    onehot[[i for i in range(len(a))], a] = 1.0
    if mask is not None:
        onehot = onehot[:, mask]
    onehot = torch.min(onehot.sum(0) / divide, torch.ones(1)).unsqueeze(0)
    # onehot = onehot.sum(0).unsqueeze(0) / float(len(a))
    return onehot

# def make_model(args):
#     if args.model == 'film':
#         model = Film(args)
#     elif args.model == 'san':
#         model = San(args)
#     elif args.model == 'rn':
#         model = RelationalNetwork(args)
#     return model
# if __name__ == '__main__':
#     dict_file = os.path.join('/home/sungwon/data', 'vqa2', 'data_dict_0.pkl')
#     with open(dict_file, 'rb') as file:
#         data_dict = pickle.load(file)
#     idx_to_question_type = data_dict['idx_to_question_type']
#     print(idx_to_question_type)
