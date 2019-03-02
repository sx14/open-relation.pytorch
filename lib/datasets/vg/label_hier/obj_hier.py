import os
from nltk.corpus import wordnet as wn
from lib.datasets.label_hier import LabelHier
from lib.datasets.label_hier import LabelNode
from global_config import PROJECT_ROOT


class ObjNet(LabelHier):

    def _check_dead_node(self):
        root = self.root()
        background = self.get_node_by_name('__background__')
        c = 1
        for i in range(self.label_sum()):
            n = self.get_node_by_index(i)
            if len(n.hypers()) == 0 and n.index() != root.index() and n.index() != background.index():
                print('%d: <%s> is dead. (no parent)' % (c, n.name()))
                c += 1
            if len(n.children()) == 0 and not n.is_raw():
                print('%d: <%s> is dead. (no children)' % (c, n.name()))
                c += 1

    def _export_split(self):
        records = []
        for node in self._index2node:
            if len(node.hypers()) > 1:
                hyper_names = ' '.join([h.name() for h in node.hypers()])
                node_name = node.name()
                record = '%s|%s\n' % (node_name, hyper_names)
                records.append(record)
        file_path = os.path.join(os.path.dirname(__file__), 'splits.txt')
        with open(file_path, 'w') as f:
            f.writelines(records)

    def _import_split(self):
        file_path = os.path.join(os.path.dirname(__file__), 'splits1.txt')
        with open(file_path, 'r') as f:
            splits = f.readlines()
        return splits

    def _prone_and_check(self, splits):
        c2p = {}
        for split in splits:
            names = split.strip().split('|')
            c = names[0].strip()
            p = names[1].strip()
            c2p[c] = p

        # check and prone
        root_node = self.root()
        raw_labels = self.get_raw_labels()[1:]  # no 'background'
        for raw_label in raw_labels:
            raw_node = self.get_node_by_name(raw_label)
            curr_node = raw_node
            while curr_node.index() != root_node.index():
                if len(curr_node.hypers()) > 1:
                    next = None
                    remove = []
                    for h in curr_node.hypers():
                        if h.name() == c2p[curr_node.name()]:
                            next = h
                        else:
                            remove.append(h)
                    if next is None:
                        print('%s -> %s' % (raw_label, curr_node.name()))
                        exit(-1)
                    else:
                        for r in remove:
                            r.del_child(curr_node)
                            if len(r.children()) == 0:
                                self._index2node[r.index()] = None
                            curr_node.del_hyper(r)
                        curr_node = next

                elif len(curr_node.hypers()) == 1:
                    curr_node = curr_node.hypers()[0]
            # print('%s -> %s' % (raw_label, curr_node.name()))

    def _create_label_nodes(self, raw2wn):
        next_label_index = 1
        # except 'background'
        for raw_label in self._raw_labels[1:]:
            wn_labels = raw2wn[raw_label]
            for wn_label in wn_labels:
                wn_node = wn.synset(wn_label)
                hypernym_paths = wn_node.hypernym_paths()  # including wn_node self
                for hypernym_path in hypernym_paths:
                    for i, w in enumerate(hypernym_path):
                        node = self.get_node_by_name(w.name())
                        if node is None:
                            node = LabelNode(w.name(), next_label_index, False)
                            self._label2node[w.name()] = node
                            self._index2node.append(node)
                            next_label_index += 1
                        if i > 0:
                            node.add_hyper(last_node)
                            last_node.add_child(node)
                        last_node = node

            # raw label is unique
            raw_node = LabelNode(raw_label, next_label_index, True)
            self._label2node[raw_label] = raw_node
            self._index2node.append(raw_node)
            next_label_index += 1
            for wn_label in raw2wn[raw_label]:
                wn_node = self.get_node_by_name(wn_label)
                raw_node.add_hyper(wn_node)
                wn_node.add_child(raw_node)

    def _construct_hier(self):
        raw2wn = self._raw2wn
        self._create_label_nodes(raw2wn)
        # self._export_split()
        # TODO: annual prone
        splits = self._import_split()
        self._prone_and_check(splits)

    def _raw_to_wn(self, raw2wn_path):
        vg_labels = self._load_raw_label(raw2wn_path)
        raw2wn = dict()
        raw_labels = []
        for vg_label in vg_labels:
            raw_label, wn_labels = vg_label.split('|')
            raw_labels.append(raw_label)
            wn_labels = wn_labels.split(' ')
            raw2wn[raw_label] = wn_labels
        return raw2wn

    def __init__(self, raw_label_path, raw2wn_path):
        self._raw2wn = self._raw_to_wn(raw2wn_path)
        LabelHier.__init__(self, raw_label_path)


raw_label_path = os.path.join(PROJECT_ROOT, 'data', 'VGdevkit2007', 'VOC2007', 'object_labels.txt')
raw2wn_path = os.path.join(PROJECT_ROOT, 'data', 'VGdevkit2007', 'VOC2007', 'object_label2wn.txt')
objnet = ObjNet(raw_label_path, raw2wn_path)

