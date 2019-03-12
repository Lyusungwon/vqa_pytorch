import torch.nn as nn
from models.default import Default


class San(nn.Module, Default):
    def __init__(self, args):
        super().__init__()
        self.init_encoders(args)
        self.cv_fc = nn.Linear(args.cv_filter, args.san_k)
        self.te_fc = nn.Linear(args.te_hidden, args.san_k)
        self.blocks = nn.ModuleList([SanBlock(args.san_k, args.san_k) for _ in range(args.san_layer)])
        self.fc = nn.Linear(args.san_k, args.a_size)

    def forward(self, image, question, question_length):
        x = self.visual_encoder(image)
        b, c, h, w = x.size()
        x = x.view(b, c, -1).transpose(2, 1)
        _, u = self.text_encoder(question, question_length)
        x = self.cv_fc(x).transpose(2, 1)
        u = self.te_fc(u)
        for block in self.blocks:
            u = block(x, u)
        logits = self.fc(u)
        return logits


class SanBlock(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.wia = nn.Linear(d, k, bias=False)
        self.wqa = nn.Linear(d, k)
        self.wp = nn.Linear(k, 1)

    def forward(self, i, q):
        wi = self.wia(i.transpose(2, 1))
        wq = self.wqa(q).unsqueeze(1)
        ha = torch.tanh(wi + wq)
        pi = torch.softmax(self.wp(ha), dim=1)
        u = torch.matmul(i, pi).squeeze(2) + q
        return u
