import os
import pickle
import numpy as np
from nltk.corpus import wordnet as wn
from open_relation import global_config
from open_relation.dataset.dataset_config import DatasetConfig



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
            print('%s -> %s' % (hypo.name(), hyper.name()))
            hypernyms.append([hypo.index(), hyper.index()])
            nodes.insert(0, hypo)

    # save hypernym dataset
    hypernyms = np.array(hypernyms)
    import h5py
    f = h5py.File(hypernym_save_path, 'w')
    f.create_dataset('hypernyms', data=hypernyms)
    f.close()


if __name__ == '__main__':

    dataset = 'vrd'
    data_config = DatasetConfig(dataset)

    if dataset == 'vrd':
        from open_relation.dataset.vrd.label_hier.obj_hier import objnet
    else:
        from open_relation.dataset.vg.label_hier.obj_hier import objnet

    label2index = objnet.label2index()

    hypernym_save_path = os.path.join(global_config.project_root,
                                      'open_relation', 'label_embedding', 'object',
                                      dataset+'_dataset', 'wordnet_with_'+dataset+'.h5')
    generate_direct_hypernyms(objnet, hypernym_save_path)
