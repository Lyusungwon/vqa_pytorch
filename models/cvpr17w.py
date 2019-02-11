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
        self.fa = Gated_Tanh(args.te_embedding + filters, args.te_embedding)
        self.wa = nn.Linear(args.te_embedding, 1)
        self.fq = Gated_Tanh(args.te_embedding, embed)
        self.fv = Gated_Tanh(filters, embed)



    def forward(self, image, question, question_length):
        i = image if self.cv_pretrained else self.visual_encoder(image)
        b, c, h, w = i.size()
        i = i.view(b, c, -1).transpose(1, 2)  # b o c
        _, q = self.text_encoder(question, question_length).unsqueeze(1)  # b 1 q
        w = torch.softmax(self.wa(self.fa(torch.cat([i, q], 2))))
        ai = torch.matmul(i, w)
        h = self.fq(q) * self.fv(ai)


        q

        i1 = self.Vf(i)  # b o h
        q1 = self.Uq(q).unsqueeze(1)  # b 1 h
        f = self.P1(i1 * q1).transpose(1, 2)  # b g o
        i2 = torch.matmul(f, i1).view(b, -1)  # b g*h
        i3 = self.Vv(i2)  # b h
        q2 = self.Wq(q)  # b h
        logits = self.P2(i3 * q2).squeeze(1)  # b o
        return logits

def attention(objects, code):

    x_coordinate = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(n, 1, h, w).to(device)
    y_coordinate = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(n, 1, h, w).to(device)
    coordinate_encoded = torch.cat([objects, x_coordinate, y_coordinate], 1)
    question = code.view(n, hd, 1, 1).expand(n, hd, h, w)
    question_encoded = torch.cat([coordinate_encoded, question], 1).view(n, -1, o).transpose(1, 2)
    return coordinate_encoded.view(n, -1, o).transpose(1, 2), question_encoded


class Gated_Tanh(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()
        self.fx = nn.Linear(inp, oup)
        self.fy = nn.Linear(inp, oup)

    def forward(self, x):
        yt = torch.tanh(self.fy(x))
        g = torch.sigmoid(self.fx(x))
        return g * yt
