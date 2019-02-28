import os
PROJECT_ROOT = os.path.dirname(__file__)


class HierLabelConfig:
    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target

    def label_vec_path(self):
        return os.path.join(PROJECT_ROOT,
                            'hier_label',
                            self.target,
                            'label_vec_%s.h5' % self.dataset)

    def direct_relation_path(self):
        return os.path.join(PROJECT_ROOT,
                            'hier_label',
                            self.target,
                            '%s_dataset' % self.dataset,
                            'wordnet_with_%s.h5' % self.dataset)