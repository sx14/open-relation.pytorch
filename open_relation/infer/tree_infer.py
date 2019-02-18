# -*- coding: utf-8 -*-
import numpy as np
import sys


class LabelPath:
    def __init__(self, label_inds, label_weights, labels):
        self._labels = labels
        self._label_num = len(label_inds)
        self._label_inds = label_inds
        self._depth_scores = label_weights
        self._label2rank = dict()
        self._scores = [-1] * len(label_inds)
        for label_ind in label_inds:
            self._label2rank[label_ind] = -1

    def cal_scores(self, rank_scores):
        if len(self._label2rank) < len(self._label_inds):
            return None
        if self._scores[0] < 0:
            for i, label_ind in enumerate(self._label_inds):
                depth_score = self._depth_scores[i]
                rank_score = rank_scores[self._label2rank[label_ind]]
                self._scores[i] = depth_score * rank_score
        return self._scores

    def try_hit(self, label_ind, label_rank):
        """
        try to hit label in current path
        :param label_ind: target label index
        :param label_rank: label rank
        :return: current hit ratio
        """
        if label_ind in self._label2rank:
            self._label2rank[label_ind] = label_rank
            return True
        else:
            return False

    def get_pred(self):
        np_scores = np.array(self._scores)
        p_ind = np.argmax(np_scores)
        pred = self._label_inds[p_ind]
        return pred, self._label2rank[pred]+1

    def get_avg_score(self):
        return np.array(self._scores).mean()

    def get_org_rank(self):
        return self._label2rank[self._label_inds[-1]]

    def get_leaf_label_ind(self):
        return self._label_inds[-1]


def cal_rank_scores(label_num):
    # rank scores [1 - 10]
    # s = a(x - b)^2 + c
    # if rank is 0, score is 10
    # b = num-1
    s_min = 1.0
    s_max = 10.0
    b = label_num - 1
    c = s_min
    a = (s_max - c) / b ** 2
    rank_scores = [0] * label_num
    for r in range(label_num):
        rank_scores[r] = a*(r-b)**2 + c
    return rank_scores

def cal_rank_scores1(n_item):
    s_max = 10
    ranks = np.arange(1, n_item+1).astype(np.float)

    s = (np.cos(ranks / n_item * np.pi) + 1) * (s_max * 1.0 / 2)
    return s


def my_infer(scores, org2path, org2pw, label2index, index2label, rank_scores):
    index2label = np.array(index2label)

    # all label paths
    all_paths = [LabelPath(org2path[org_ind], org2pw[org_ind], index2label[org2path[org_ind]]) for org_ind in org2path]

    # label_ind 2 rank
    ind2ranks = [0] * len(label2index.keys())
    ranked_inds = np.argsort(scores).tolist()
    ranked_inds.reverse()  # descending

    for rank in range(len(ranked_inds)):
        # rank start from 1
        ind2ranks[ranked_inds[rank]] = rank
        # try to fill all paths
        for path in all_paths:
            path.try_hit(ranked_inds[rank], rank)

    path_scores = []
    org_ranks = []
    for path in all_paths:
        org_ranks.append(path.get_org_rank())
        path.cal_scores(rank_scores)
        path_scores.append(path.get_avg_score())

    pred_path_ind = np.argmax(np.array(path_scores))
    pred_path = all_paths[pred_path_ind]
    pred_ind, pred_rank = pred_path.get_pred()

    cand_path_ind = np.argmin(np.array(org_ranks))
    cand_ind = all_paths[cand_path_ind].get_leaf_label_ind()
    cand_rank = np.min(np.array(org_ranks))+1

    cands = [[pred_ind, pred_rank], [cand_ind, cand_rank]]

    return pred_ind, cands
