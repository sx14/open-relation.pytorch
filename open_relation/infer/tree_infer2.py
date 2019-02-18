# -*- coding: utf-8 -*-
import sys
import numpy as np





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


class TreeNode:
    def __init__(self, name, index):
        self._rank = -1
        self._name = name
        self._index = index
        self._parents = []
        self._children = []

    def __str__(self):
        return '%s[%d]' % (self._name, self._rank)

    def add_children(self, child):
        self._children.append(child)

    def children(self):
        return self._children

    def append_parent(self, parent):
        self._parents.append(parent)

    def set_rank(self, rank):
        self._rank = rank

    def rank(self):
        return self._rank

    def index(self):
        return self._index

    def name(self):
        return self._name


def construct_tree(label_hier, ranked_inds, ind2rank):
    ind2node = dict()
    for label in label_hier.get_all_labels():
        hnode = label_hier.get_node_by_name(label)
        tnode = TreeNode(label, hnode.index())
        ind2node[hnode.index()] = tnode

    for label in label_hier.get_all_labels():
        hnode = label_hier.get_node_by_name(label)
        tnode = ind2node[hnode.index()]
        hypers = hnode.hypers()
        for hyper in hypers:
            pnode = ind2node[hyper.index()]
            pnode.add_children(tnode)
            tnode.append_parent(pnode)

    for r, ind in enumerate(ranked_inds):
        rank = r + 1  # 1 based
        ind2rank[ind] = rank
        tnode = ind2node[ind]
        tnode.set_rank(rank)

    return ind2node


def top_down(tree, label_hier):
    def choose_child(children, parent_rank):
        choice = None
        if len(children) == 1:
            choice = children[0]
        elif len(children) > 1:
            ranked_children = sorted(children, key=lambda c: c.rank())
            r1 = ranked_children[0].rank()
            r2 = ranked_children[1].rank()
            if (r2 - r1) > r1:
                # r1 is confident, and doesn't confuse with r2
                choice = ranked_children[0]
        return choice

    # root as default
    root_ind = label_hier.root().index()
    tnode = tree[root_ind]
    while tnode:
        choice = tnode
        tnode = choose_child(tnode.children(), 0)
    return [choice.index(), choice.rank()]


def bottom_up(tree, label_hier, top2_raw, thr):
    node1 = label_hier.get_node_by_index(top2_raw[0][0])
    node2 = label_hier.get_node_by_index(top2_raw[1][0])
    n1_path = node1.trans_hyper_inds()
    n2_path = node2.trans_hyper_inds()
    min_plength = min(len(n1_path), len(n2_path))
    common_path = set(n1_path) & set(n2_path)
    if len(common_path) * 1.0 / min_plength >= thr:
        pred_ind = max(common_path)
        return [pred_ind, tree[pred_ind].rank()]
    else:
        return top2_raw[0]


def my_infer(label_hier, scores, rank_scores, target):
    obj_thr = {'b_u': 0.75,
               't_d': 0.5,
               'min_dis': label_hier.label_sum() / 7,
               'half': label_hier.label_sum() / 3}
    pre_thr = {'b_u': 0.6,
               't_d': 0.4,
               'min_dis': 3,
               'half': 10}
    thr = {'obj': obj_thr,
           'pre': pre_thr}

    threshold = thr[target]

    # label_ind 2 rank
    ind2rank = [0] * label_hier.label_sum()
    ranked_inds = np.argsort(scores).tolist()
    ranked_inds.reverse()  # descending

    # top2 raw label as default predictions
    raw_top2 = []
    for r, ind in enumerate(ranked_inds):
        if label_hier.get_node_by_index(ind).is_raw() and len(raw_top2) < 2:
            raw_top2.append([ind, r+1])

    # confident part
    half_rank = threshold['half']
    if raw_top2[0][1] < half_rank and (raw_top2[1][1] - raw_top2[0][1]) > threshold['min_dis']:
        cands = raw_top2
    elif raw_top2[0][1] < half_rank and (raw_top2[1][1] - raw_top2[0][1]) <= threshold['min_dis']:
        ind2node = construct_tree(label_hier, ranked_inds, ind2rank)
        cands = [bottom_up(ind2node, label_hier, raw_top2, threshold['b_u']), raw_top2[0]]
    elif raw_top2[0][1] >= half_rank and (raw_top2[1][1] - raw_top2[0][1]) <= threshold['min_dis']:
        ind2node = construct_tree(label_hier, ranked_inds, ind2rank)
        cands = [bottom_up(ind2node, label_hier, raw_top2, threshold['t_d']), raw_top2[0]]
        if cands[0][0] == raw_top2[0][0]:
            cands = [top_down(ind2node, label_hier), raw_top2[0]]
    else:
        ind2node = construct_tree(label_hier, ranked_inds, ind2rank)
        cands = [top_down(ind2node, label_hier), raw_top2[0]]
    return cands

