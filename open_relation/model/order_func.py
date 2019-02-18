import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


def order_sim(hypers, hypos):
    sub = hypers - hypos
    act = nn.functional.relu(sub)
    partial_order_dis = act.norm(p=2, dim=1)
    partial_order_sim = -partial_order_dis
    return partial_order_sim


def order_softmax_test(batch_scores, pos_neg_inds, punish):
    punish_v = Variable(torch.from_numpy(np.array(punish))).float().cuda()

    loss_scores = Variable(torch.zeros(len(batch_scores), len(pos_neg_inds[0]))).float().cuda()
    for i in range(len(batch_scores)):
        scores = batch_scores[i] * punish_v
        loss_scores[i] = scores[pos_neg_inds[i]]

        # loss_scores[i] = batch_scores[i, pos_neg_inds[i]]
    y = Variable(torch.zeros(len(batch_scores))).long().cuda()
    acc = 0.0
    for scores in loss_scores:
        p_score = scores[0]
        n_score_max = torch.max(scores[1:])
        if p_score > n_score_max:
            acc += 1
    acc = acc / len(batch_scores)
    return acc, loss_scores, y