import torch
import torch.nn.functional as F
from layers import *
from utils import *


class CVPR17W(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cv_pretrained = args.cv_pretrained
        pretrained_weight = load_pretrained_embedding(args.i2q, args.te_embedding) if args.te_pretrained else None
        self.text_encoder = TextEncoder(args.q_size, args.te_embedding, args.te_type, args.te_hidden, args.te_layer, args.te_dropout, pretrained_weight)
        if args.cv_pretrained:
            filters = 2048 if args.dataset == 'vqa2' else 1024
        else:
            self.visual_encoder = Conv(args.cv_filter, args.cv_kernel, args.cv_stride, args.cv_layer, args.cv_batchnorm)
            filters = args.cv_filter
        self.fa = Gated_Tanh(args.te_hidden * 2 + filters, args.te_hidden * 2)
        self.wa = nn.Linear(args.te_hidden * 2, 1)
        self.fq = Gated_Tanh(args.te_hidden * 2, args.c17w_hidden)
        self.fv = Gated_Tanh(filters, args.c17w_hidden)
        if args.cf_pretrained:
            self.fo = Gated_Tanh(args.c17w_hidden, args.te_embedding)
            pretrained_weight = load_pretrained_embedding(args.i2a, args.te_embedding)
            self.wo = nn.Linear(args.te_embedding, args.a_size)
            self.wo.weight.data.copy_(pretrained_weight)
        else:
            self.fo = Gated_Tanh(args.c17w_hidden, args.c17w_hidden)
            self.wo = nn.Linear(args.c17w_hidden, args.a_size)
            # self.do = nn.Dropout(args.c17w_dropout, inplace=True)

    def forward(self, image, question, question_length):
        i = image if self.cv_pretrained else self.visual_encoder(image)
        b, c, h, w = i.size()
        o = h * w
        i = i.view(b, c, -1).transpose(1, 2)  # b o c
        # i = F.normalize(i, -1)
        _, q = self.text_encoder(question, question_length)  # b q
        qa = q.unsqueeze(1).expand(-1, o, -1) # b o q
        aw = torch.softmax(self.wa(self.fa(torch.cat([i, qa], 2))), dim=1).transpose(1, 2)
        ai = torch.bmm(aw, i).squeeze(1)
        h = self.fq(q) * self.fv(ai)
        logits = self.wo(self.fo(h))
        # logits = self.do(self.wo(self.fo(h)))
        return logits


class Gated_Tanh(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()
        self.fx = nn.Linear(inp, oup)
        self.fy = nn.Linear(inp, oup)

    def forward(self, x):
        yt = torch.tanh(self.fy(x))
        g = torch.sigmoid(self.fx(x))
        return g * yt
