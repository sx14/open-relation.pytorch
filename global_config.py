import os
PROJECT_ROOT = os.path.dirname(__file__)


def label_vec_path(dataset):
    return os.path.join(PROJECT_ROOT,
                        'hier_label',
                        'object',
                        'label_vec_%s.h5' % dataset)