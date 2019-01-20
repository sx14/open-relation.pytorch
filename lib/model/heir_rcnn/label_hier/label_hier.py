import os


class LabelNode(object):
    def __init__(self, name, index):
        self._weight = -1
        self._index = index
        self._name = name
        self._hypers = []

    def __str__(self):
        return self._name

    def name(self):
        return self._name

    def hypers(self):
        return self._hypers

    def append_hyper(self, hyper):
        self._hypers.append(hyper)

    def index(self):
        return self._index

    def trans_hyper_inds(self):
        hyper_inds = []
        unique_hyper_inds = set()
        h_paths = self.hyper_paths()
        for p in h_paths:
            for w in p:
                if w.index() not in unique_hyper_inds:
                    unique_hyper_inds.add(w.index())
                    hyper_inds.append(w.index())
        return hyper_inds

    def hyper_paths(self):
        if len(self._hypers) == 0:
            # root
            return [[self]]
        else:
            paths = []
            for hyper in self._hypers:
                sub_paths = hyper.hyper_paths()
                for sub_path in sub_paths:
                    sub_path.append(self)
                    paths.append(sub_path)
            return paths

    def show_hyper_paths(self):
        paths = self.hyper_paths()
        for p in paths:
            str = []
            for n in p:
                str.append(n._name)
                str.append('->')
            str = str[:-1]
            print(' '.join(str))


class LabelHier:

    def label2index(self):
        l2i = dict()
        for l in self._label2node:
            l2i[l] = self._label2node[l].index()
        return l2i

    def index2label(self):
        i2l = []
        for n in self._index2node:
            i2l.append(n.name())
        return i2l

    def raw2path(self):
        r2p = dict()
        for r in self._raw_labels:
            rn = self.get_node_by_name(r)
            r2p[r] = rn.trans_hyper_inds()
        return r2p

    def label_sum(self):
        return len(self._index2node)

    def get_all_labels(self):
        return sorted(self._label2node.keys())

    def get_raw_labels(self):
        return self._raw_labels

    def get_node_by_name(self, name):
        if name in self._label2node:
            return self._label2node[name]
        else:
            return None

    def get_node_by_index(self, index):
        if index < len(self._index2node):
            return self._index2node[index]
        else:
            return None

    def _load_raw_label(self, raw_label_path):
        labels = []
        if os.path.exists(raw_label_path):
            with open(raw_label_path, 'r') as f:
                raw_lines = f.readlines()
                for line in raw_lines:
                    labels.append(line.strip('\n'))
        else:
            print('Raw label file not exists !')
        return labels

    def _construct_hier(self):
        raise NotImplementedError

    def __init__(self, pre_label_path):
        self._raw_labels = self._load_raw_label(pre_label_path)
        # self._raw_labels.insert(0, '__background__')
        bk = LabelNode('__background__', 0)
        self._label2node = {'__background__': bk}
        self._index2node = [bk]
        self._construct_hier()