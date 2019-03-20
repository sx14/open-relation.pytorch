# -*- coding: utf-8 -*-
from math import exp, log
import numpy as np

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn.functional import softmax


class TreeNode:
    def __init__(self, name, index, info_ratio):
        self._cond_prob = {
            'vis': 0.0,
            'lan': 0.0
        }
        self._raw_score = {
            'vis': 0.0,
            'lan': 0.0
        }
        self._prob = {
            'vis': 0.0,
            'lan': 0.0
        }
        self._name = name
        self._index = index
        self._parents = []
        self._children = []
        self._info_ratio = info_ratio

    def __str__(self):
        return '%s[%.2f]' % (self._name, self.score())

    def raw_score(self, type):
        return self._raw_score[type]

    def set_raw_score(self, type, raw_score):
        self._raw_score[type] = raw_score

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

    def set_cond_prob(self, type, cond_prob):
        self._cond_prob[type] = cond_prob

    def cond_prob(self, type):
        return self._cond_prob[type]

    def prob(self, type):
        if self._prob is None:
            p = self.cond_prob(type)
            curr = self
            while len(curr._parents) > 0:
                p *= curr._parents[0].cond_prob(type)
                curr = curr._parents[0]
            self._prob = p
        return self._prob

    def info_ratio(self):
        return self._info_ratio

    def score(self, type):
        if self._raw_score[type] <= -1:
            scr = self._raw_score[type] + 1
        else:
            scr = -1.0 / min(self._raw_score[type], -0.0001) - 1
        scr = 1.0 / (1.0 + exp(-scr))
        return scr

    def entropy(self, type):
        e = 0.0
        for c in self._children:
            e -= c.cond_prob(type) * log(c.cond_prob(type), len(self._children))
        return min(e, 1.0)

    def index(self):
        return self._index

    def name(self):
        return self._name


def construct_tree(labelnet, vis_scores, lan_scores):
    ind2node = []
    # node meta info
    for ind in range(labelnet.label_sum()):
        hnode = labelnet.get_node_by_index(ind)
        tnode = TreeNode(hnode.name(), ind, hnode.depth_ratio())
        tnode.set_raw_score('vis', vis_scores[ind].tolist())
        tnode.set_raw_score('lan', lan_scores[ind].tolist())
        ind2node.append(tnode)

    # node hierarchy
    for label in labelnet.get_all_labels():
        hnode = labelnet.get_node_by_name(label)
        tnode = ind2node[hnode.index()]
        hypers = hnode.hypers()
        for hyper in hypers:
            pnode = ind2node[hyper.index()]
            pnode.add_children(tnode)
            tnode.append_parent(pnode)

    return ind2node


def top_down_search(root):
    root.set_cond_prob(1.0)
    node = root
    path_scores = [0.0]
    path_nodes = [root]
    # print('P0\t\tP1\t\tE\t\tI')
    while len(node.children()) > 0:
        c_vis_scores = []
        c_lan_scores = []
        for c in node.children():
            c_vis_scores.append(c.raw_score('vis'))
            c_lan_scores.append(c.raw_score('lan'))
        c_vis_scores_v = Variable(Tensor(c_vis_scores))
        c_lan_scores_v = Variable(Tensor(c_lan_scores))
        c_vis_scores_s = softmax(c_vis_scores_v, 0)
        c_lan_scores_s = softmax(c_lan_scores_v, 0)

        for i, c in enumerate(node.children()):
            c.set_cond_prob('vis', c_vis_scores_s[i].data.numpy().tolist())
            c.set_cond_prob('lan', c_lan_scores_s[i].data.numpy().tolist())

        if node.entropy('vis') > 0.7 and node.entropy('lan') > 0.7 and node.depth() > 2:
            break

        if node.entropy('vis') - node.entropy('lan') < -0.1:
            scr_type_use = 'vis'
            c_scores_use = c_vis_scores_s
        else:
            scr_type_use = 'lan'
            c_scores_use = c_lan_scores_s

        pred_c_ind = torch.argmax(c_scores_use)
        pred_c_node = node.children()[pred_c_ind]
        hedge_c_scr = pred_c_node.info_ratio() * (1 - node.entropy(scr_type_use))
        path_scores.append(hedge_c_scr)
        path_nodes.append(pred_c_node)
        # print('(%.2f)\t(%.2f)\t(%.2f)\t(%.2f) %s' % (node.prob(), node.cond_prob(), node.entropy(), node.info_ratio(), node.name()))
        node = pred_c_node

    # for i in range(len(path_nodes)):
        # print('%s: %.2f' % (path_nodes[i].name(), path_scores[i]))

    max_scr_ind = np.argmax(np.array(path_scores))
    max_scr_node = path_nodes[max_scr_ind]

    return max_scr_node




def my_infer(labelnet, vis_scores, lan_scores):
    tnodes = construct_tree(labelnet, vis_scores, lan_scores)
    choice = top_down_search(tnodes[labelnet.root().index()])
    return [[choice.index(), choice.score()], [choice.index(), choice.score()]]


