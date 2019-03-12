import torch
import torch.nn as nn
from layers import MLP, Conv
from models.default import Default
from torch.nn.init import kaiming_uniform_


class RelationalNetwork(nn.Module, Default):
    def __init__(self, args):
        super(RelationalNetwork, self).__init__()
        self.init_encoders(args)
        self.g_theta = MLP((args.cv_filter + 2) * 2 + args.te_hidden,
                           args.rn_gt_hidden,
                           args.rn_gt_hidden,
                           args.rn_gt_layer)
        self.f_phi = MLP(args.rn_gt_hidden,
                         args.rn_fp_hidden,
                         args.a_size,
                         args.rn_fp_layer,
                         args.rn_fp_dropout,
                         last=True)
        if args.cv_pretrained:
            self.visual_encoder = nn.Sequential(
                                    nn.Conv2d(1024, args.cv_filter, 3, 2, padding=1),
                                    nn.BatchNorm2d(args.cv_filter),
                                    nn.ReLU()
                                    # nn.Conv2d(args.cv_filter, args.cv_filter, 3, 2, padding=1),
                                    # nn.BatchNorm2d(args.cv_filter),
                                    # nn.ReLU()
            )
            self.init()

    def forward(self, image, question, question_length):
        x = self.visual_encoder(image)
        _, code = self.text_encoder(question, question_length)
        pairs = rn_encode(x, code)
        relations = self.g_theta(pairs).sum(1).squeeze()
        # relations = lower_sum(relations)
        logits = self.f_phi(relations)
        return logits

    def init(self):
        for i in self.visual_encoder:
            if isinstance(i, nn.Conv2d):
                kaiming_uniform_(i.weight)


def rn_encode(images, questions):
    try:
        device = images.get_device()
    except:
        device = torch.device('cpu')
    n, c, h, w = images.size()
    o = h * w
    hd = questions.size(1)
    x_coordinate = torch.linspace(-h/2, h/2, h).view(1, h, 1, 1).expand(n, h, w, 1).contiguous().view(n, o, 1).to(device)
    y_coordinate = torch.linspace(-w/2, w/2, w).view(1, 1, w, 1).expand(n, h, w, 1).contiguous().view(n, o, 1).to(device)
    images = images.view(n, c, o).transpose(1, 2)
    images = torch.cat([images, x_coordinate, y_coordinate], 2)
    images1 = images.unsqueeze(1).expand(n, o, o, c + 2).contiguous()
    images2 = images.unsqueeze(2).expand(n, o, o, c + 2).contiguous()
    questions = questions.unsqueeze(1).unsqueeze(2).expand(n, o, o, hd)
    pairs = torch.cat([images1, images2, questions], 3).view(n, o**2, -1)
    # pairs = torch.cat([images1, images2, questions], 3)
    return pairs


def lower_sum(relations):
    try:
        device = relations.get_device()
    except:
        device = torch.device('cpu')
    n, h, w, l = relations.size()
    mask = torch.ones([h, w]).tril().view(1, h, w, 1).to(device, dtype=torch.uint8)
    relations = torch.masked_select(relations, mask).view(n, -1, l)
    return relations.sum(1)
