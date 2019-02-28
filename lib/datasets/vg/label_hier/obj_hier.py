import os
from nltk.corpus import wordnet as wn
from open_relation.dataset.label_hier import LabelHier
from open_relation.dataset.label_hier import LabelNode
from open_relation.dataset.dataset_config import DatasetConfig


class ObjNet(LabelHier):

    def raw2wn(self):
        return self._raw2wn

    def _create_label_nodes(self, raw2wn):
        # keep wn label unique
        wn_label_set = set()
        next_label_index = 1
        for raw_label in self._raw_labels:
            wn_labels = raw2wn[raw_label]
            for wn_label in wn_labels:
                if wn_label not in wn_label_set:
                    wn_node = wn.synset(wn_label)
                    hypernym_paths = wn_node.hypernym_paths()   # including wn_node self
                    for hypernym_path in hypernym_paths:
                        for w in hypernym_path:
                            if w.name() not in wn_label_set:
                                wn_label_set.add(w.name())
                                # create new label node
                                node = LabelNode(w.name(), next_label_index, False)
                                self._label2node[w.name()] = node
                                self._index2node.append(node)
                                next_label_index += 1

            # raw label is unique
            node = LabelNode(raw_label, next_label_index, True)
            self._label2node[raw_label] = node
            self._index2node.append(node)
            next_label_index += 1

    def _construct_hier(self):
        raw2wn = self._raw2wn
        self._create_label_nodes(raw2wn)
        # link the nodes
        for i in range(1, len(self._index2node)):
            node = self._index2node[i]
            if node.name() in raw2wn:
                # raw label node
                hyper_wn_label = raw2wn[node.name()][0]
                node.append_hyper(self._label2node[hyper_wn_label])
            else:
                # wn label node
                w = wn.synset(node.name())
                for h in w.hypernyms() + w.instance_hypernyms():
                    if h.name() in self._label2node:
                        node.append_hyper(self._label2node[h.name()])

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


dataset_config = DatasetConfig('vg')
raw_label_path = os.path.join(dataset_config.dataset_root, 'object_labels.txt')
raw2wn_path = os.path.join(dataset_config.dataset_root, 'object_label2wn.txt')
objnet = ObjNet(raw_label_path, raw2wn_path)

# if __name__ == '__main__':
#     a = ObjNet(raw_label_path, raw2wn_path)
#     n = a.get_node_by_name('road')
#     n.show_hyper_paths()