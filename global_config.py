import os

PROJECT_ROOT = os.path.dirname(__file__)
VG_ROOT = os.path.join(PROJECT_ROOT, 'data', 'VGdevkit2007', 'VOC2007')
VRD_ROOT = os.path.join(PROJECT_ROOT, 'data', 'VRDdevkit2007', 'VOC2007')


class HierLabelConfig:
    def __init__(self, dataset, target):
        # dataset = vrd, vg
        self.dataset = dataset
        # target = object predicate
        self.target = target

    def label_vec_path(self):
        return os.path.join(PROJECT_ROOT,
                            'hier_label',
                            'label_vec_%s_%s.h5'
                            % (self.dataset, self.target))

