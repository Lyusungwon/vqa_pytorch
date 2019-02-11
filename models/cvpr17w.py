import torch
from layers import *
from utils import *


class CVPR17W(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cv_pretrained = args.cv_pretrained
        pretrained_weight = load_pretrained_embedding(args.word_to_idx, args.te_embedding) if args.te_pretrained else None
        self.text_encoder = TextEncoder(args.q_size, args.te_embedding, args.te_type, args.te_hidden, args.te_layer, args.te_dropout, pretrained_weight)
        if args.cv_pretrained:
            filters = 2048 if args.dataset == 'vqa2' else 1024
        else:
            self.visual_encoder = Conv(args.cv_filter, args.cv_kernel, args.cv_stride, args.cv_layer, args.cv_batchnorm)
            filters = args.cv_filter
        self.fa = Gated_Tanh(args.te_hidden + filters, args.te_hidden)
        self.wa = nn.Linear(args.te_hidden, 1)
        self.fq = Gated_Tanh(args.te_hidden, args.c17w_hidden)
        self.fv = Gated_Tanh(filters, args.c17w_hidden)
        self.fo = Gated_Tanh(args.c17w_hidden, args.c17w_hidden)
        self.wo = nn.Linear(args.c17w_hidden, args.a_size)

    def forward(self, image, question, question_length):
        i = image if self.cv_pretrained else self.visual_encoder(image)
        b, c, h, w = i.size()
        o = h * w
        i = i.view(b, c, -1).transpose(1, 2)  # b o c
        _, q = self.text_encoder(question, question_length)  # b q
        qe = q.unsqueeze(1).expand(-1, o, -1) # b o q
        aw = torch.softmax(self.wa(self.fa(torch.cat([i, qe], 2))), dim=1).transpose(1, 2)
        ai = torch.matmul(aw, i).squeeze(1)
        h = self.fq(q) * self.fv(ai)
        logits = torch.sigmoid(self.wo(self.fo(h)))
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
