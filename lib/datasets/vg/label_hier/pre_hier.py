import os
from lib.datasets.label_hier import LabelHier
from lib.datasets.label_hier import LabelNode
from global_config import PROJECT_ROOT

class PreNet(LabelHier):

    def _construct_hier(self):
        # root node
        # 0 is background
        # TODO: construct predicate label hierarchy
        pass

    def __init__(self, raw_label_path):
        LabelHier.__init__(self, raw_label_path)


label_path = os.path.join(PROJECT_ROOT, 'data', 'VRDdevkit2007', 'VOC2007', 'predicate_labels.txt')
prenet = PreNet(label_path)

# if __name__ == '__main__':
#     a = PreNet()
#     n = a.get_pre('stand next to')
#     n.show_hyper_paths()