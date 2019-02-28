import os
import global_config


def label_vec_path(dataset):
    return os.path.join(global_config.PROJECT_ROOT,
                        'hier_label',
                        'object',
                        'label_vec_%s.h5' % dataset)
