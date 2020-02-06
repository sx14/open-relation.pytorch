import torch
import numpy as np
from torch.autograd import Variable
from torch.nn.functional import cross_entropy

class OrderLoss:
    def __init__(self, labelnet):
        self.labelnet = labelnet
        self.n_neg_classes = labelnet.neg_num()

    # prepare labels [pos, neg1, neg2, ...]
    def _loss_labels(self, rois_label):
        loss_labels = np.zeros((rois_label.size()[0], 1+self.n_neg_classes)).astype(np.int)
        for i, gt_ind in enumerate(rois_label):
            gt_label = self.labelnet.get_all_labels()[int(gt_ind)]
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
