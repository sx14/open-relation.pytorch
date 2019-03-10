# -*- coding: utf-8 -*-
from math import exp, log
import numpy as np

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn.functional import softmax


class TreeNode:
    def __init__(self, name, index, info_ratio):
        self._cond_prob = 0.0
        self._raw_score = None
        self._prob = 0.0
        self._name = name
        self._index = index
        self._parents = []
        self._children = []
        self._info_ratio = info_ratio

    def __str__(self):
        return '%s[%.2f]' % (self._name, self.score())

    def raw_score(self):
        return self._raw_score

    def set_raw_score(self, raw_score):
        self._raw_score = raw_score

    def depth(self):
        min_p_depth = 0
        for p in self._parents:
            min_p_depth = max(min_p_depth, p.depth())
        return min_p_depth + 1

    def add_children(self, child):
        self._children.append(child)

    def children(self):
        return self._children

    def append_parent(self, parent):
        self._parents.append(parent)

    def set_cond_prob(self, cond_prob):
        self._cond_prob = cond_prob

    def cond_prob(self):
        return self._cond_prob

    def prob(self):
        if self._prob is None:
            p = self.cond_prob()
            curr = self
            while len(curr._parents) > 0:
                p *= curr._parents[0].cond_prob()
                curr = curr._parents[0]
            self._prob = p
        return p

    def info_ratio(self):
        return self._info_ratio

    def score(self):
        return self.cond_prob() * self._info_ratio

    def entropy(self):
        e = 0.0
        for c in self._children:
            e -= c.cond_prob() * log(c.cond_prob(), len(self._children))
        return e

    def index(self):
        return self._index

    def name(self):
        return self._name



def construct_tree(labelnet, scores):
    ind2node = []
    # node meta info
    for ind in range(labelnet.label_sum()):
        hnode = labelnet.get_node_by_index(ind)
        tnode = TreeNode(hnode.name(), ind, hnode.info_ratio(labelnet.pos_leaf_sum()))
        ind2node.append(tnode)

    # for label in labelnet.get_all_labels():
    #     hnode = labelnet.get_node_by_name(label)
    #     tnode = TreeNode(label, hnode.index(), hnode.info_ratio(labelnet.pos_leaf_sum()))
    #     ind2node[hnode.index()] = tnode

    # node hierarchy
    for label in labelnet.get_all_labels():
        hnode = labelnet.get_node_by_name(label)
        tnode = ind2node[hnode.index()]
        hypers = hnode.hypers()
        for hyper in hypers:
            pnode = ind2node[hyper.index()]
            pnode.add_children(tnode)
            tnode.append_parent(pnode)

    # node raw score
    ranked_inds = np.argsort(scores)[::-1]
    for r, ind in enumerate(ranked_inds):
        score = scores[ind]
        tnode = ind2node[ind]
        tnode.set_raw_score(score.tolist())

    return ind2node


def depth_thresh(depth):
    return min(exp(depth - 10), 0.9)


def entropy_thresh(scores):
    entropy = 0
    for score in scores:
        entropy += (-log(score, 2)) * score


def good_thresh(max_depth, depth):
    depth_ratio = -max(max_depth/2.0-depth, 0)/(max_depth/2.0)
    thresh = 0.6 * exp(depth_ratio)
    return thresh


def top_down_search(root, max_depth, threshold=0):
    root.set_cond_prob(1.0)
    node = root
    print('P0\t\tP1\t\tE\t\tI')
    while len(node.children()) > 0:
        c_scores = []
        for c in node.children():
            c_scores.append(c.raw_score())
        c_scores_v = Variable(Tensor(c_scores))
        c_scores_s = softmax(c_scores_v, 0)

        for i, c in enumerate(node.children()):
            c.set_cond_prob(c_scores_s[i].data.numpy().tolist())

        pred_c_ind = torch.argmax(c_scores_s)
        pred_c_scr = c_scores_s[pred_c_ind]

        threshold = good_thresh(node.depth(), max_depth)
        if pred_c_scr < threshold:
            break

        print('(%.2f)\t(%.2f)\t(%.2f)\t(%.2f) %s' % (node.prob(), node.cond_prob(), node.entropy(), node.info_ratio(), node.name()))

        if node.entropy() >= 1:
            a = 1

        node = node.children()[pred_c_ind]
    print('(%.2f)\t(%.2f)\t(%.2f)\t(%.2f) %s' % (node.prob(), node.cond_prob(), node.entropy(), node.info_ratio(), node.name()))
    return node


def cal_pos_cond_prob(node):
    if len(node.children()) == 0:
        return

    c_scores = []
    for c in node.children():
        c_scores.append(c.raw_score())
    c_scores_v = Variable(Tensor(c_scores))
    c_scores_s = softmax(c_scores_v, 0)
    for i, c in enumerate(node.children()):
        c.set_cond_prob(c_scores_s[i].data.numpy().tolist())

    pred_c_ind = torch.argmax(c_scores_s)
    cal_pos_cond_prob(node.children()[pred_c_ind])




def my_infer(labelnet, scores):
    tnodes = construct_tree(labelnet, scores)
    choice = top_down_search(tnodes[labelnet.root().index()], 0.4)
    return [[choice.index(), choice.score()], [choice.index(), choice.score()]]


def raw2cond_prob(labelnet, batch_scores):

    cond_probs = np.zeros(batch_scores.shape)
    for i in range(batch_scores.shape[0]):
        scores = batch_scores[i]
        tnodes = construct_tree(labelnet, scores)
        root = tnodes[labelnet.root().index()]
        root.set_cond_prob(1.0)
        cal_pos_cond_prob(root)

        for j in range(batch_scores.shape[1]):
            cond_probs[i][j] = tnodes[j].cond_prob()

    return cond_probs

