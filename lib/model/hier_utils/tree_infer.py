# -*- coding: utf-8 -*-
from math import exp, log
import numpy as np

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn.functional import softmax


class TreeNode:
    def __init__(self, name, index, info_ratio):
        self._score = -1
        self._name = name
        self._index = index
        self._parents = []
        self._children = []
        self._info_ratio = info_ratio

    def __str__(self):
        return '%s[%.2f]' % (self._name, self._score)

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

    def set_cond_prob(self, score):
        self._score = score

    def cond_prob(self):
        return self._score

    def prob(self):
        p = self.cond_prob()
        curr = self
        while len(curr._parents) > 0:
            p *= curr._parents[0].cond_prob()
        return p

    def score(self):
        return self.prob() * self._info_ratio

    def index(self):
        return self._index

    def name(self):
        return self._name



def construct_tree(labelnet, scores):
    ind2node = dict()
    for label in labelnet.get_all_labels():
        hnode = labelnet.get_node_by_name(label)
        tnode = TreeNode(label, hnode.index(), hnode.info_ratio(labelnet.pos_leaf_sum()))
        ind2node[hnode.index()] = tnode

    for label in labelnet.get_all_labels():
        hnode = labelnet.get_node_by_name(label)
        tnode = ind2node[hnode.index()]
        hypers = hnode.hypers()
        for hyper in hypers:
            pnode = ind2node[hyper.index()]
            pnode.add_children(tnode)
            tnode.append_parent(pnode)

    ranked_inds = np.argsort(scores)[::-1]
    for r, ind in enumerate(ranked_inds):
        # rank = r + 1.0  # 1 based
        # score = (len(ranked_inds) - rank) / len(ranked_inds)
        score = scores[ind]
        tnode = ind2node[ind]
        tnode.set_cond_prob(score.tolist())

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
    node = root
    root.set_cond_prob(1.0)
    while len(node.children()) > 0:
        c_scores = []
        for c in node.children():
            c_scores.append(c.cond_prob())
        c_scores_v = Variable(Tensor(c_scores))
        c_scores_s = softmax(c_scores_v, 0)
        pred_c_ind = torch.argmax(c_scores_s)
        pred_c_scr = c_scores_s[pred_c_ind]

        threshold = good_thresh(node.depth(), max_depth)
        if pred_c_scr < threshold:
            break
        node = node.children()[pred_c_ind]
        node.set_cond_prob(pred_c_scr.data.numpy().tolist())
    return node


def my_infer(labelnet, scores):
    tnodes = construct_tree(labelnet, scores)
    choice = top_down_search(tnodes[labelnet.root().index()], 0.4)
    return [[choice.index(), choice.score()], [choice.index(), choice.score()]]
