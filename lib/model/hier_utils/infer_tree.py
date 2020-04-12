# -*- coding: utf-8 -*-
from math import exp, log
import numpy as np

from torch import Tensor
from torch.autograd import Variable
from torch.nn.functional import softmax

'''
Class InferTree is used to find top k predictions from the given scores of each concept.
You can refer to the section of Greedy Inference.
'''


class TreeNode:
    def __init__(self, name, index, info_ratio, is_raw):
        self._cond_prob = 0.0
        self._raw_score = None
        self._max_descent_score = None
        self._prob = 0.0
        self._name = name
        self._index = index
        self._parents = []
        self._children = []
        self._info_ratio = info_ratio
        self._is_raw = is_raw
        self._used = False

    def is_used(self):
        return self._used

    def set_used(self):
        self._used = True

    def __str__(self):
        return '%s[%.2f]' % (self._name, self.score())

    def is_raw(self):
        return self._is_raw

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
        return self._prob

    def info_ratio(self):
        return self._info_ratio

    def score(self):
        log_score = -log(max(0.0001, -self._raw_score))
        scr = 1.0 / (1.0 + exp(-log_score))
        return scr

    def entropy(self):

        if len(self._children) > 1:
            e = 0.0
            for c in self._children:
                e -= c.cond_prob() * log(c.cond_prob(), len(self._children))
            return min(e, 1.0)
        else:
            return 1.0

    def index(self):
        return self._index

    def name(self):
        return self._name

    def parents(self):
        return self._parents

    def max_descent_score(self):
        return self._max_descent_score

    def set_max_descent_score(self):
        if len(self.children()) == 0:
            self._max_descent_score = self._raw_score
        else:
            children_scores = []
            for c in self.children():
                c.set_max_descent_score()
                score = c.max_descent_score()
                children_scores.append(score)

            self._max_descent_score = max(np.max(np.array(children_scores)),
                                          self._raw_score)


class InferTree:
    def __init__(self, labelnet, scores):
        self.__ind2node = []

        # node meta info
        for ind in range(labelnet.label_sum()):
            hnode = labelnet.get_node_by_index(ind)
            tnode = TreeNode(hnode.name(), ind, hnode.depth_ratio(), hnode.is_raw())
            self.__ind2node.append(tnode)

        # node hierarchy
        for label in labelnet.get_all_labels():
            hnode = labelnet.get_node_by_name(label)
            tnode = self.__ind2node[hnode.index()]
            hypers = hnode.hypers()
            for hyper in hypers:
                pnode = self.__ind2node[hyper.index()]
                pnode.add_children(tnode)
                tnode.append_parent(pnode)

        ranked_inds = np.argsort(scores)[::-1]
        for r, ind in enumerate(ranked_inds):
            score = scores[ind]
            tnode = self.__ind2node[ind]
            tnode.set_raw_score(score.tolist())

        self.__root = self.__ind2node[labelnet.root().index()]
        self.__root.set_max_descent_score()

    def top_k(self, k, mode='hier'):
        assert k > 0, 'k should be larger than zero'
        top_k = []
        for t in range(k):
            choice = self.__top_down_search(self.__root, mode)
            top_k.append([choice.index(), choice.score()])
        return top_k

    def __top_down_search(self, root, mode='hier'):
        root.set_cond_prob(1.0)
        node = root
        path_scores = [0.0]
        path_nodes = [root]
        while len(node.children()) > 0:
            c_scores = []
            for c in node.children():
                c_scores.append(c.max_descent_score())
            c_scores_v = Variable(Tensor(c_scores))
            c_scores_s = softmax(c_scores_v, 0)

            for i, c in enumerate(node.children()):
                c.set_cond_prob(c_scores_s[i].data.numpy().tolist())

            inds = np.argsort(c_scores_s.numpy())[::-1]
            pred_c_node = None
            for ind in inds:
                cand = node.children()[ind]
                if cand.is_used():
                    continue
                else:
                    pred_c_node = cand
                    break
            assert pred_c_node is not None
            hedge_c_scr = pred_c_node.info_ratio() * (1 - node.entropy())

            node.set_used()
            node = pred_c_node
            if mode == 'raw' and not pred_c_node.is_raw():
                continue

            path_scores.append(hedge_c_scr)
            path_nodes.append(pred_c_node)

        max_scr_ind = np.argmax(np.array(path_scores))
        max_scr_node = path_nodes[max_scr_ind]

        return max_scr_node
