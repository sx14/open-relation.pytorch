import os
from lib.datasets.label_hier import LabelHier
from lib.datasets.label_hier import LabelNode
from global_config import PROJECT_ROOT


class PreNet(LabelHier):

    def _construct_hier(self):
        # STUB IMPLEMENTATION

        # 0 is background
        # root node
        next_index = 1
        root = LabelNode('relation', next_index, False)
        for raw_label in self._raw_labels:
            next_index += 1
            node = LabelNode(raw_label, next_index, True)
            node.add_hyper(root)
            root.add_child(node)
            self._index2node.append(node)
            self._label2node[node.name()] = node
        # TODO: construct predicate label hierarchy

    def __init__(self, raw_label_path):
        LabelHier.__init__(self, raw_label_path)


label_path = os.path.join(PROJECT_ROOT, 'data', 'VGdevkit2007', 'VOC2007', 'predicate_labels.txt')
prenet = PreNet(label_path)
