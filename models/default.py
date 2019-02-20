from layers import *
from utils import load_pretrained_embedding
from torch.nn.init import kaiming_uniform


class Default:
    # def __init__(self, args):
    def init_encoders(self, args):
        self.cv_pretrained = args.cv_pretrained
        pretrained_weight = load_pretrained_embedding(args.i2q, args.te_embedding) if args.te_pretrained else None
        self.text_encoder = TextEncoder(args.q_size,
                                        args.te_embedding,
                                        args.te_type,
                                        args.te_hidden,
                                        args.te_layer,
                                        args.te_dropout,
                                        args.te_bidir,
                                        pretrained_weight)
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
