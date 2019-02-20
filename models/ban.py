import torch
import torch.nn.functional as F
from layers import *
from utils import *


class BAN(nn.Module):
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

        self.attention_block = BANBlock(args.te_hidden, filters, args.ban_hidden, args.ban_glimpse)
        self.blocks = nn.ModuleList([BANBlock() for i in range(args.ban_glimpse)])
        self.U = nn.Sequential(
            nn.Linear(args.te_hidden, args.ban_hidden * args.ban_k),
            nn.ReLU(inplace=True)
        )
        self.V = nn.Sequential(
            nn.Linear(filters, args.ban_hidden * args.ban_k),
            nn.ReLU(inplace=True)
        )
        self.h = nn.Linear(args.ban_hidden * args.ban_k, args.ban_hout)

        self.classifier = MLP()

    def forward(self, image, question, question_length):
        i = image if self.cv_pretrained else self.visual_encoder(image)
        b, c, h, w = i.size()
        i = i.view(b, c, -1).transpose(1, 2).transpose(1, 2)  # b d o1
        q, _ = self.text_encoder(question, question_length)
        q = q[0].transpose(1, 2) # b d o2
        aw = self.attention_block(i, q)


        b_emb = [0] * self.ban_glimpse
        for n, block in enumerate(self.blocks):
            b_emb[n] = block.forward_with_weights(i, q, aw[:, n, :, :])
            q = q +


        d = torch.matmul(i.unsqueeze(3), q.unsqueeze(2)) # b d o2 o3
        d = self.h(d.transpose(1, 2).transpose(2, 3)) # b o2 o3 d
        return d.transpose(1, 2).transpose(2, 3)  # b o2 o3 h
        # i = F.normalize(i, -1)



        qa = q.unsqueeze(1).expand(-1, o, -1) # b o q
        aw = torch.softmax(self.wa(self.fa(torch.cat([i, qa], 2))), dim=1).transpose(1, 2)
        ai = torch.bmm(aw, i).squeeze(1)
        h = self.fq(q) * self.fv(ai)
        logits = self.wo(self.fo(h))
        # logits = self.do(self.wo(self.fo(h)))
        return logits


class BANBlock(nn.Module):
    def __init__(self, q, i, d, o):
        super().__init__()
        self.U = nn.Sequential(
            nn.Linear(q, d),
            nn.ReLU(inplace=True)
        )
        self.V = nn.Sequential(
            nn.Linear(i, d),
            nn.ReLU(inplace=True)
        )
        self.h = nn.Linear(d, o)

    def forward(self, i, q, w):
        i = self.V(i).transpose(1, 2).unsqueeze(2)  # b d 1 o1
        q = self.U(q).transpose(1, 2).unsqueeze(3)  # b d o2 1
        w = w.unsqueeze(1) # b 1 o1 o2
        logits = torch.matmul(torch.matmul(i, w), q) # b d 1 1
        return logits.squeeze(3).squeeze(2) # b d

class BANAttBlock(nn.Module):
    def __init__(self, q, i, d, g, k):
        self.q = q
        self.i = i
        self.d = d
        self.g = g
        self.k = k
        super().__init__()
        self.U = nn.Sequential(
            nn.Linear(q, d),
            nn.ReLU(inplace=True)
        )
        self.V = nn.Sequential(
            nn.Linear(i, d),
            nn.ReLU(inplace=True)
        )
        self.glimpse = nn.Linear(d, g)
        self.p = nn.AvgPool1d(k, stride=k)

    def forward(self, i, q):
        # i -> b oi i
        i_ = self.V(i).transpose(1, 2)  # b d o1
        q_ = self.U(q).transpose(1, 2)  # b d o2
        d = torch.matmul(i_.unsqueeze(3), q_.unsqueeze(2))  # b d o1 o2
        d = self.h(d.transpose(1, 2).transpose(2, 3)).transpose(1, 2).transpose(2, 3)  # b o o1 o2
        aw = torch.softmax(d.view(-1, self.g, self.i * self.q), 2).view(-1, self.g, self.i, self.q) # b o o1 o2
        ai = torch.bmm(aw, i).squeeze(1)

        return d  # b o o1 o2

