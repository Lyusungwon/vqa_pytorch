from layers import *
from utils import load_pretrained_embedding, load_bert
from torch.nn.init import kaiming_uniform
import torch

class Default:
    # def __init__(self, args):
    def init_encoders(self, args):
        self.cv_pretrained = args.cv_pretrained
        if not args.te_bert:
            pretrained_weight = load_pretrained_embedding(args.i2q, args.te_embedding) if args.te_pretrained else None
            self.text_encoder = TextEncoder(args.q_size,
                                            args.te_embedding,
                                            args.te_type,
                                            args.te_hidden,
                                            args.te_layer,
                                            args.te_dropout,
                                            args.te_bidir,
                                            pretrained_weight)
            if args.te_bidir:
                args.te_hidden = args.te_hidden * 2
        else:
            self.text_encoder = BertEncoder(args.te_hidden)
        if args.cv_pretrained:
            filters = 2048 if args.dataset == 'vqa2' else 1024
            self.visual_encoder = nn.Conv2d(filters, args.cv_filter, 3, 1)
            kaiming_uniform(self.visual_encoder.weight)
            self.visual_encoder.bias.data.zero_()
        else:
            self.visual_encoder = Conv(args.cv_filter,
                                       args.cv_kernel,
                                       args.cv_stride,
                                       args.cv_layer,
                                       args.cv_batchnorm)


class BertEncoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.text_encoder = load_bert(latent_size)

    def forward(self, question, question_len):
        try:
            device = question.get_device()
        except:
            device = torch.device('cpu')
        segment_ids = torch.zeros_like(question).to(device)
        out = self.text_encoder(question, segment_ids)
        return None, out