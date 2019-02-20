import torch
import torch.nn as nn
from layers import MLP
from models.default import Default
from torch.nn.init import kaiming_uniform_


class Film(nn.Module, Default):
    def __init__(self, args):
        super(Film, self).__init__()
        self.init_encoders(args)
        self.filters = args.cv_filter
        self.layers = args.film_res_layer
        self.fc = nn.Linear(args.te_hidden, args.cv_filter * args.film_res_layer * 2)
        self.res_blocks = nn.ModuleList([FilmResBlock(args.cv_filter, args.film_res_kernel) for _ in range(args.film_res_layer)])
        self.classifier = FilmClassifier(args.cv_filter, args.film_cf_filter, args.film_fc_hidden, args.a_size, args.film_fc_layer)
        self.init()

    def forward(self, image, question, question_length):
        if not self.cv_pretrained:
            image = image * 2 - 1
        x = self.visual_encoder(image)
        _, code = self.text_encoder(question, question_length)
        betagamma = self.fc(code).view(-1, self.layers, 2, self.filters)
        for n, block in enumerate(self.res_blocks):
            x = block(x, betagamma[:, n])
        logits = self.classifier(x)
        return logits

    def init(self):
        kaiming_uniform_(self.fc.weight)
        self.fc.bias.data.zero_()


class FilmResBlock(nn.Module):
    def __init__(self, filter, kernel):
        super(FilmResBlock, self).__init__()
        self.conv1 = nn.Conv2d(filter + 2, filter, 1, 1, 0)
        self.conv2 = nn.Conv2d(filter, filter, kernel, 1, (kernel - 1)//2, bias=False)
        self.batch_norm = nn.BatchNorm2d(filter)
        self.relu = nn.ReLU(inplace=True)
        self.init()

    def forward(self, x, betagamma):
        x = positional_encode(x)
        x = self.relu(self.conv1(x))
        residual = x
        beta = betagamma[:, 0].unsqueeze(2).unsqueeze(3).expand_as(x)
        gamma = betagamma[:, 1].unsqueeze(2).unsqueeze(3).expand_as(x)
        x = self.batch_norm(self.conv2(x))
        x = self.relu(x * beta + gamma)
        x = x + residual
        return x

    def init(self):
        kaiming_uniform_(self.conv1.weight)
        self.conv1.bias.data.zero_()
        kaiming_uniform_(self.conv2.weight)


class FilmClassifier(nn.Module):
    def __init__(self, filter, last_filter, hidden, last, layer):
        super(FilmClassifier, self).__init__()
        self.conv = nn.Conv2d(filter + 2, last_filter, 1, 1, 0)
        # self.pool = nn.MaxPool2d((input_h, input_w))
        self.mlp = MLP(last_filter, hidden, last, layer)
        self.init()

    def forward(self, x):
        x = positional_encode(x)
        x = self.conv(x).max(2)[0].max(2)[0]
        x = self.mlp(x)
        return x

    def init(self):
        kaiming_uniform_(self.conv.weight)
        self.conv.bias.data.zero_()


def positional_encode(images):
    try:
        device = images.get_device()
    except:
        device = torch.device('cpu')
    n, c, h, w = images.size()
    x_coordinate = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(n, 1, h, w).to(device)
    y_coordinate = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(n, 1, h, w).to(device)
    images = torch.cat([images, x_coordinate, y_coordinate], 1)
    return images
