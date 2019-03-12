from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import kaiming_uniform_


class TextEncoder(nn.Module):
    def __init__(self,
                 vocab,
                 embedding,
                 type,
                 hidden,
                 num_layer=1,
                 dropout=0,
                 bidirectional=False,
                 pretrained_weight=None):
        super(TextEncoder, self).__init__()
        self.type = type
        self.embedding = nn.Embedding(vocab, embedding, padding_idx=0)
        if type == 'gru':
            self.seq = nn.GRU(embedding, hidden, num_layers=num_layer, dropout=dropout, bidirectional=bidirectional,  batch_first=True)
        elif type == 'lstm':
            self.seq = nn.LSTM(embedding, hidden, num_layers=num_layer, dropout=dropout, bidirectional=bidirectional,  batch_first=True)
        # self.gru.flatten_parameters()
        self.init(pretrained_weight)
        print(self.embedding)
        print(self.seq)

    def forward(self, question, question_length):
        b_size = question.size()[0]
        embedded = self.embedding(question)
        packed_embedded = pack_padded_sequence(embedded, question_length, batch_first=True)

        self.seq.flatten_parameters()
        if self.type == 'gru':
            output, h_n = self.seq(packed_embedded)
        elif self.type == 'lstm':
            output, (h_n, c_n) = self.seq(packed_embedded)
        h_n = h_n.permute(1, 0, 2).contiguous().view(b_size, -1)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output, h_n

    def init(self, pretrained_weight):
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(pretrained_weight)
        else:
            self.embedding.weight.data.uniform_(-0.1, 0.1)
            # self.embedding.weight.require_grad = False


class Conv(nn.Module):
    def __init__(self,
                 filter,
                 kernel,
                 stride,
                 layer,
                 batchnorm):
        super(Conv, self).__init__()
        self.batchnorm = batchnorm
        prev_filter = 3
        net = nn.ModuleList([])
        for _ in range(layer - 1):
            net.append(nn.Conv2d(prev_filter, filter, kernel, stride, (kernel - 1)//2, bias=not batchnorm))
            if batchnorm:
                net.append(nn.BatchNorm2d(filter))
            net.append(nn.ReLU(inplace=True))
            prev_filter = filter
        self.net = nn.Sequential(*net)
        self.init()
        print(self.net)

    def forward(self, x):
        x = self.net(x)
        return x

    def init(self):
        for i in self.net:
            if isinstance(i, nn.Conv2d):
                kaiming_uniform_(i.weight)
                if not self.batchnorm:
                    i.bias.data.zero_()


class MLP(nn.Module):
    def __init__(self,
                 input,
                 hidden,
                 output,
                 layer,
                 dropout=None,
                 last=False):
        super(MLP, self).__init__()
        layers = [input] + [hidden for _ in range(layer - 1)] + [output]
        net = []
        for n, (inp, outp) in enumerate(zip(layers, layers[1:])):
            net.append(nn.Linear(inp, outp))
            net.append(nn.ReLU(inplace=True))
            if dropout and n == layer - 2:
                # net.insert(-3, nn.Dropout(dropout))
                net.append(nn.Dropout(dropout))
        if last:
            net = net[:-1]
        net = nn.ModuleList(net)
        self.net = nn.Sequential(*net)
        self.init()
        print(self.net)

    def forward(self, x):
        x = self.net(x)
        return x

    def init(self):
        for i in self.net:
            if isinstance(i, nn.Linear):
                kaiming_uniform_(i.weight)
                i.bias.data.zero_()
