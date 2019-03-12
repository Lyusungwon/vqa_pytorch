import torch.nn as nn
from models.default import Default


class Mrn(nn.Module, Default):
    def __init__(self, args):
        super().__init__()
        self.init_encoders(args)
        self.first_block = MrnBlock(args.cv_filter, args.te_hidden, args.mrn_hidden)
        self.blocks = nn.ModuleList([MrnBlock(args.cv_filter, args.mrn_hidden, args.mrn_hidden) for _ in range(args.mrn_layer - 1)])
        self.fc = nn.Linear(args.mrn_hidden, args.a_size)

    def forward(self, image, question, question_length):
        x = self.visual_encoder(image).mean(3).mean(2)
        _, q = self.text_encoder(question, question_length)
        h = self.first_block(q, x)
        for block in self.blocks:
            h = block(h, x)
        logits = self.fc(h)
        return logits


class MrnBlock(nn.Module):
    def __init__(self, i, q, h):
        super().__init__()
        self.qs = nn.Sequential(
            nn.Linear(q, h),
            nn.Tanh()
        )
        self.vb = nn.Sequential(
            nn.Linear(i, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh()
        )
        self.res = nn.Linear(q, h)

    def forward(self, q, i):
        question = self.qs(q)
        objects = self.vb(i)
        h = objects * question + self.res(q)
        return h
