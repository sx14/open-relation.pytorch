import numpy as np

import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.autograd import Variable


class OrderSimilarity(nn.Module):
    def __init__(self, norm):
        super(OrderSimilarity, self).__init__()
        self._norm = norm
        self.act = nn.ReLU()

    def forward1(self, lab_vecs, vis_vecs):
        order_scores = Variable(torch.zeros(vis_vecs.size()[0], lab_vecs.size()[0])).cuda()
        for i in range(vis_vecs.size()[0]):
            # hyper - hypo
            sub = lab_vecs - vis_vecs[i]
            # max(sub, 0)
            sub = self.act(sub)
            # norm 2
            order_dis = sub.norm(p=self._norm, dim=1)
            order_sim = -order_dis
            order_scores[i] = order_sim
        return order_scores

    def forward(self, lab_vecs, vis_vecs):
        d_vec = lab_vecs.size(1)
        n_label = lab_vecs.size(0)
        n_vis = vis_vecs.size(0)

        stack_lab_vecs = lab_vecs.repeat(n_vis, 1)
        stack_vis_vecs = vis_vecs.repeat(1, n_label).reshape(n_vis * n_label, d_vec)

        stack_sub = stack_lab_vecs - stack_vis_vecs
        stack_sub = self.act(stack_sub)
        stack_dis = stack_sub.norm(p=self._norm, dim=1)
        stack_sim = - stack_dis
        order_sims = stack_sim.reshape(n_vis, n_label)
        return order_sims


class OrderLoss:
    def __init__(self, labelnet):
        self.labelnet = labelnet
        self.n_neg_classes = labelnet.neg_num()

    # prepare labels [pos, neg1, neg2, ...]
    def _loss_labels(self, rois_label):
        loss_labels = np.zeros((rois_label.size()[0], 1+self.n_neg_classes)).astype(np.int)
        for i, gt_ind in enumerate(rois_label):
            gt_label = self.labelnet.get_all_labels()[gt_ind]
            gt_node = self.labelnet.get_node_by_name(gt_label)
            all_pos_inds = set(gt_node.trans_hyper_inds())
            all_neg_inds = list(set(range(self.labelnet.label_sum())) - all_pos_inds)
            loss_labels[i] = [rois_label[i]] + all_neg_inds[:self.n_neg_classes]
        return loss_labels

    # prepare cls scores [pos, neg1, neg2, ...]
    def _prepare_loss_input(self, cls_scores, pos_negs):
        loss_scores = Variable(torch.zeros(len(cls_scores), len(pos_negs[0]))).float().cuda()
        for i in range(len(cls_scores)):
            scores = cls_scores[i]
            loss_scores[i] = scores[pos_negs[i]]
        y = Variable(torch.zeros(len(loss_scores))).long().cuda()
        return loss_scores, y

    def forward(self, cls_scores, labels):
        pos_negs = self._loss_labels(labels)
        cls_scores_use, y = self._prepare_loss_input(cls_scores, pos_negs)
        return cross_entropy(cls_scores_use, y)
