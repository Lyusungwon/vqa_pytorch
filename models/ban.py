import torch
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from utils import *
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from models.default import Default


class BanModel(nn.Module, Default):
    def __init__(self, args):
        # , dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, op, glimpse
        super(BanModel, self).__init__()
        self.init_encoders(args)
        # self.dataset = args.dataset
        self.op = 'c'
        self.glimpse = args.ban_glimpse
        self.v_att = BiAttention(args.cv_filter, args.ban_hidden, args.ban_hidden, args.ban_glimpse)
        self.b_net = nn.ModuleList([])
        self.q_prj = nn.ModuleList([])
        self.c_prj = nn.ModuleList([])
        objects = 10  # minimum number of boxes
        for i in range(args.gamma):
            self.b_net.append(BCNet(args.cv_filter, args.ban_hidden, args.ban_hidden, None, k=1))
            self.q_prj.append(FCNet([args.ban_hidden, args.ban_hidden], '', .2))
            self.c_prj.append(FCNet([objects + 1, args.ban_hidden], 'ReLU', .0))
        self.classifier = SimpleClassifier(
            args.ban_hidden, args.ban_hidden * 2, args.a_size, .5)
        self.counter = Counter(objects)
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()

        # self.w_emb = w_emb
        # self.q_emb = q_emb

    # def forward(self, v, b, q, labels):
    def forward(self, image, question, question_length):
    #        self,
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        v = self.visual_encoder(image)
        # v =
        q, _ = self.text_encoder(question, question_length)

        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]
        boxes = b[:, :, :4].transpose(1, 2)

        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v, q_emb)  # b x g x v x q

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att[:, g, :, :])  # b x l x h

            atten, _ = logits[:, g, :, :].max(2)
            embed = self.counter(boxes, atten)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)

        logits = self.classifier(q_emb.sum(1))

        return logits, att


def build_ban(dataset, num_hid, op='', gamma=4):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, op)
    q_emb = QuestionEmbedding(300 if 'c' not in op else 600, num_hid, 1, False, .0)
    v_att = BiAttention(dataset.v_dim, num_hid, num_hid, gamma)
    b_net = []
    q_prj = []
    c_prj = []
    objects = 10  # minimum number of boxes
    for i in range(gamma):
        b_net.append(BCNet(dataset.v_dim, num_hid, num_hid, None, k=1))
        q_prj.append(FCNet([num_hid, num_hid], '', .2))
        c_prj.append(FCNet([objects + 1, num_hid], 'ReLU', .0))
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, .5)
    counter = Counter(objects)
    return BanModel(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, op, gamma)


class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3), \
            name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        p, logits = self.forward_all(v, q, v_mask)
        return p, logits

    def forward_all(self, v, q, v_mask=True):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v,q) # b x g x v x q

        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, -float('inf'))

        p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
        return p.view(-1, self.glimpse, v_num, q_num), logits

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if ''!=act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if ''!=act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    """

    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2, .5], k=3):
        super(BCNet, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim;
        self.q_dim = q_dim
        self.h_dim = h_dim;
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])  # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if None == h_out:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if None == self.h_out:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            logits = d_.transpose(1, 2).transpose(2, 3)  # b x v x q x h_dim
            return logits

        # broadcast Hadamard product, matrix-matrix production
        # fast computation but memory inefficient
        # epoch 1, time: 157.84
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v)).unsqueeze(1)
            q_ = self.q_net(q)
            h_ = v_ * self.h_mat  # broadcast, b x h_out x v x h_dim
            logits = torch.matmul(h_, q_.unsqueeze(1).transpose(2, 3))  # b x h_out x v x q
            logits = logits + self.h_bias
            return logits  # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
        # epoch 1, time: 304.87
        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            return logits.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v).transpose(1, 2).unsqueeze(2)  # b x d x 1 x v
        q_ = self.q_net(q).transpose(1, 2).unsqueeze(3)  # b x d x q x 1
        logits = torch.matmul(torch.matmul(v_, w.unsqueeze(1)), q_)  # b x d x 1 x 1
        logits = logits.squeeze(3).squeeze(2)
        if 1 < self.k:
            logits = logits.unsqueeze(1)  # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k  # sum-pooling
        return logits

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class Counter(nn.Module):
    """ Counting module as proposed in [1].
    Count the number of objects from a set of bounding boxes and a set of scores for each bounding box.
    This produces (self.objects + 1) number of count features.

    [1]: Yan Zhang, Jonathon Hare, Adam Prügel-Bennett: Learning to Count Objects in Natural Images for Visual Question Answering.
    https://openreview.net/forum?id=B12Js_yRb
    """

    def __init__(self, objects, already_sigmoided=False):
        super().__init__()
        self.objects = objects
        self.already_sigmoided = already_sigmoided
        self.f = nn.ModuleList([PiecewiseLin(16) for _ in range(16)])

    def forward(self, boxes, attention):
        """ Forward propagation of attention weights and bounding boxes to produce count features.
        `boxes` has to be a tensor of shape (n, 4, m) with the 4 channels containing the x and y coordinates of the top left corner and the x and y coordinates of the bottom right corner in this order.
        `attention` has to be a tensor of shape (n, m). Each value should be in [0, 1] if already_sigmoided is set to True, but there are no restrictions if already_sigmoided is set to False. This value should be close to 1 if the corresponding boundign box is relevant and close to 0 if it is not.
        n is the batch size, m is the number of bounding boxes per image.
        """
        # only care about the highest scoring object proposals
        # the ones with low score will have a low impact on the count anyway
        boxes, attention = self.filter_most_important(self.objects, boxes, attention)
        # normalise the attention weights to be in [0, 1]
        if not self.already_sigmoided:
            attention = F.sigmoid(attention)

        relevancy = self.outer_product(attention)
        distance = 1 - self.iou(boxes, boxes)

        # intra-object dedup
        score = self.f[0](relevancy) * self.f[1](distance)

        # inter-object dedup
        dedup_score = self.f[3](relevancy) * self.f[4](distance)
        dedup_per_entry, dedup_per_row = self.deduplicate(dedup_score, attention)
        score = score / dedup_per_entry

        # aggregate the score
        # can skip putting this on the diagonal since we're just summing over it anyway
        correction = self.f[0](attention * attention) / dedup_per_row
        score = score.sum(dim=2).sum(dim=1, keepdim=True) + correction.sum(dim=1, keepdim=True)
        score = (score + 1e-20).sqrt()
        one_hot = self.to_one_hot(score)

        att_conf = (self.f[5](attention) - 0.5).abs()
        dist_conf = (self.f[6](distance) - 0.5).abs()
        conf = self.f[7](att_conf.mean(dim=1, keepdim=True) + dist_conf.mean(dim=2).mean(dim=1, keepdim=True))

        return one_hot * conf

    def deduplicate(self, dedup_score, att):
        # using outer-diffs
        att_diff = self.outer_diff(att)
        score_diff = self.outer_diff(dedup_score)
        sim = self.f[2](1 - score_diff).prod(dim=1) * self.f[2](1 - att_diff)
        # similarity for each row
        row_sims = sim.sum(dim=2)
        # similarity for each entry
        all_sims = self.outer_product(row_sims)
        return all_sims, row_sims

    def to_one_hot(self, scores):
        """ Turn a bunch of non-negative scalar values into a one-hot encoding.
        E.g. with self.objects = 3, 0 -> [1 0 0 0], 2.75 -> [0 0 0.25 0.75].
        """
        # sanity check, I don't think this ever does anything (it certainly shouldn't)
        scores = scores.clamp(min=0, max=self.objects)
        # compute only on the support
        i = scores.long().data
        f = scores.frac()
        # target_l is the one-hot if the score is rounded down
        # target_r is the one-hot if the score is rounded up
        target_l = scores.data.new(i.size(0), self.objects + 1).fill_(0)
        target_r = scores.data.new(i.size(0), self.objects + 1).fill_(0)

        target_l.scatter_(dim=1, index=i.clamp(max=self.objects), value=1)
        target_r.scatter_(dim=1, index=(i + 1).clamp(max=self.objects), value=1)
        # interpolate between these with the fractional part of the score
        return (1 - f) * Variable(target_l) + f * Variable(target_r)

    def filter_most_important(self, n, boxes, attention):
        """ Only keep top-n object proposals, scored by attention weight """
        attention, idx = attention.topk(n, dim=1, sorted=False)
        idx = idx.unsqueeze(dim=1).expand(boxes.size(0), boxes.size(1), idx.size(1))
        boxes = boxes.gather(2, idx)
        return boxes, attention

    def outer(self, x):
        size = tuple(x.size()) + (x.size()[-1],)
        a = x.unsqueeze(dim=-1).expand(*size)
        b = x.unsqueeze(dim=-2).expand(*size)
        return a, b

    def outer_product(self, x):
        # Y_ij = x_i * x_j
        a, b = self.outer(x)
        return a * b

    def outer_diff(self, x):
        # like outer products, except taking the absolute difference instead
        # Y_ij = | x_i - x_j |
        a, b = self.outer(x)
        return (a - b).abs()

    def iou(self, a, b):
        # this is just the usual way to IoU from bounding boxes
        inter = self.intersection(a, b)
        area_a = self.area(a).unsqueeze(2).expand_as(inter)
        area_b = self.area(b).unsqueeze(1).expand_as(inter)
        return inter / (area_a + area_b - inter + 1e-12)

    def area(self, box):
        x = (box[:, 2, :] - box[:, 0, :]).clamp(min=0)
        y = (box[:, 3, :] - box[:, 1, :]).clamp(min=0)
        return x * y

    def intersection(self, a, b):
        size = (a.size(0), 2, a.size(2), b.size(2))
        min_point = torch.max(
            a[:, :2, :].unsqueeze(dim=3).expand(*size),
            b[:, :2, :].unsqueeze(dim=2).expand(*size),
        )
        max_point = torch.min(
            a[:, 2:, :].unsqueeze(dim=3).expand(*size),
            b[:, 2:, :].unsqueeze(dim=2).expand(*size),
        )
        inter = (max_point - min_point).clamp(min=0)
        area = inter[:, 0, :, :] * inter[:, 1, :, :]
        return area


class PiecewiseLin(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.weight = nn.Parameter(torch.ones(n + 1))
        # the first weight here is always 0 with a 0 gradient
        self.weight.data[0] = 0

    def forward(self, x):
        # all weights are positive -> function is monotonically increasing
        w = self.weight.abs()
        # make weights sum to one -> f(1) = 1
        w = w / w.sum()
        w = w.view([self.n + 1] + [1] * x.dim())
        # keep cumulative sum for O(1) time complexity
        csum = w.cumsum(dim=0)
        csum = csum.expand((self.n + 1,) + tuple(x.size()))
        w = w.expand_as(csum)

        # figure out which part of the function the input lies on
        y = self.n * x.unsqueeze(0)
        idx = Variable(y.long().data)
        f = y.frac()

        # contribution of the linear parts left of the input
        x = csum.gather(0, idx.clamp(max=self.n))
        # contribution within the linear segment the input falls into
        x = x + f * w.gather(0, (idx + 1).clamp(max=self.n))
        return x.squeeze(0)

class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, ntoken, emb_dim, dropout, op=''):
        super(WordEmbedding, self).__init__()
        self.op = op
        self.emb = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
        if 'c' in op:
            self.emb_ = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
            self.emb_.weight.requires_grad = False # fixed
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file, tfidf=None, tfidf_weights=None):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init
        if tfidf is not None:
            if 0 < tfidf_weights.size:
                weight_init = torch.cat([weight_init, torch.from_numpy(tfidf_weights)], 0)
            weight_init = tfidf.matmul(weight_init) # (N x N') x (N', F)
            self.emb_.weight.requires_grad = True
        if 'c' in self.op:
            self.emb_.weight.data[:self.ntoken] = weight_init.clone()

    def forward(self, x):
        emb = self.emb(x)
        if 'c' in self.op:
            emb = torch.cat((emb, self.emb_(x)), 2)
        emb = self.dropout(emb)
        return emb


class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU if rnn_type == 'GRU' else None

        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        output, hidden = self.rnn(x, hidden)

        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        output, hidden = self.rnn(x, hidden)
        return output
