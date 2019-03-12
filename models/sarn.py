import torch
from torch import nn
from layers import MLP
import torch.nn.functional as F
from models.default import Default


class Sarn(nn.Module, Default):
    def __init__(self, args):
        super(Sarn, self).__init__()
        self.init_encoders(args)
        self.h_psi = MLP(args.cv_filter + 2 + args.te_hidden,
                         args.sarn_hp_hidden,
                         1,
                         args.sarn_hp_layer,
                         last=True)
        self.g_theta = MLP((args.cv_filter + 2) * 2 + args.te_hidden,
                           args.sarn_gt_hidden,
                           args.sarn_gt_hidden,
                           args.sarn_gt_layer)
        self.f_phi = MLP(args.sarn_gt_hidden,
                         args.sarn_fp_hidden,
                         args.a_size,
                         args.sarn_fp_layer,
                         args.sarn_fp_dropout,
                         last=True)

    def forward(self, image, question, question_length):
        x = self.visual_encoder(image)
        _, code = self.text_encoder(question, question_length)
        coordinate_encoded, question_encoded = sarn_encode(x, code)
        attention = self.h_psi(question_encoded)
        pairs = sarn_pair(coordinate_encoded, question_encoded, attention)
        relations = self.g_theta(pairs).sum(1)
        logits= self.f_phi(relations)
        return logits


def sarn_encode(objects, code):
    try:
        device = objects.get_device()
    except:
        device = torch.device('cpu')
    n, c, h, w = objects.size()
    o = h * w
    hd = code.size(1)
    x_coordinate = torch.linspace(-h/2, h/2, h).view(1, 1, h, 1).expand(n, 1, h, w).to(device)
    y_coordinate = torch.linspace(-w/2, w/2, w).view(1, 1, 1, w).expand(n, 1, h, w).to(device)
    coordinate_encoded = torch.cat([objects, x_coordinate, y_coordinate], 1)
    question = code.view(n, hd, 1, 1).expand(n, hd, h, w)
    question_encoded = torch.cat([coordinate_encoded, question], 1).view(n, -1, o).transpose(1, 2)
    return coordinate_encoded.view(n, -1, o).transpose(1, 2), question_encoded


def sarn_pair(coordinate_encoded, question_encoded, attention):
    selection = F.softmax(attention.squeeze(2), dim=1)
    selected = torch.bmm(selection.unsqueeze(1), coordinate_encoded).expand_as(coordinate_encoded)
    pairs = torch.cat([question_encoded, selected], 2)
    return pairs
