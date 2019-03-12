import torch
from torch import nn
from layers import MLP
from models.default import Default


class BaseRN(nn.Module, Default):
    def __init__(self, args):
        super(BaseRN, self).__init__()
        self.init_encoders(args)
        self.g_theta = MLP(args.cv_filter + 2 + args.te_hidden,
                           args.basern_gt_hidden,
                           args.basern_gt_hidden,
                           args.basern_gt_layer)
        self.f_phi = MLP(args.basern_gt_hidden,
                         args.basern_fp_hidden,
                         args.a_size,
                         args.basern_fp_layer,
                         args.basern_fp_dropout,
                         last=True)

    def forward(self, image, question, question_length):
        x = self.visual_encoder(image)
        _, code = self.text_encoder(question, question_length)
        pairs = baseline_encode(x, code)
        relations = self.g_theta(pairs).sum(1)
        logits = self.f_phi(relations)
        return logits


def baseline_encode(images, questions):
    try:
        device = images.get_device()
    except:
        device = torch.device('cpu')
    n, c, h, w = images.size()
    o = h * w
    hd = questions.size(1)
    x_coordinate = torch.linspace(-h/2, h/2, h).view(1, 1, h, 1).expand(n, 1, h, w).to(device)
    y_coordinate = torch.linspace(-w/2, w/2, w).view(1, 1, 1, w).expand(n, 1, h, w).to(device)
    questions = questions.unsqueeze(2).unsqueeze(3).expand(n, hd, h, w)
    images = torch.cat([images, x_coordinate, y_coordinate, questions], 1).view(n, -1, o).transpose(1, 2)
    return images
