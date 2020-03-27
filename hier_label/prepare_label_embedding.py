import os
import pickle
import numpy as np
from nltk.corpus import wordnet as wn
from global_config import HierLabelConfig



def generate_direct_hypernyms(labelnet, hypernym_save_path):
    # ==== generate direct hypernym relations ====
    # [[hypo , hyper]]
    hypernyms = []
    # BFS
    nodes = [labelnet.root()]
    while len(nodes) > 0:
        hyper = nodes.pop()
        hypos = hyper.children()
        for hypo in hypos:
            # print('%s -> %s' % (hypo.name(), hyper.name()))
            hypernyms.append([hypo.index(), hyper.index()])
            nodes.insert(0, hypo)

    # save hypernym dataset
    hypernyms = np.array(hypernyms)
    import h5py
    f = h5py.File(hypernym_save_path, 'w')
    f.create_dataset('hypernyms', data=hypernyms)
    f.close()


if __name__ == '__main__':

    dataset = 'vglsj'
    target = 'predicate'
    if dataset == 'vrd':
        from lib.datasets.vrd.label_hier.obj_hier import objnet
        from lib.datasets.vrd.label_hier.pre_hier import prenet
    else:
        from lib.datasets.vg200.label_hier.obj_hier import objnet
        from lib.datasets.vg200.label_hier.pre_hier import prenet

    if target == 'object':
        labelnet = objnet
    else:
        labelnet = prenet

    label2index = labelnet.label2index()

    hypernym_save_path = '%s_dataset/hypernym_%s.h5' % (dataset, target)
    generate_direct_hypernyms(labelnet, hypernym_save_path)
