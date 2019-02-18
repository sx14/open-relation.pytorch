import os
import pickle
import numpy as np
from nltk.corpus import wordnet as wn
from open_relation import global_config
from open_relation.dataset.dataset_config import DatasetConfig



def generate_direct_hypernyms(raw2wn, label2index, hypernym_save_path):
    # ==== generate direct hypernym relations ====
    hypernyms = []
    all_wn_nodes = set()
    # [[hypo, hyper]]
    for raw_label in raw2wn:
        wn_labels = raw2wn[raw_label]
        for wn_label in wn_labels:
            # vg_label -> wn_label
            hypernyms.append([label2index[raw_label], label2index[wn_label]])
            wn_node = wn.synset(wn_label)
            for wn_path in wn_node.hypernym_paths():
                for w in wn_path:
                    all_wn_nodes.add(w)
    for wn_node in all_wn_nodes:
        for h in wn_node.hypernyms() + wn_node.instance_hypernyms():
            if h.name() in label2index:
                hypernyms.append([label2index[wn_node.name()], label2index[h.name()]])
    # save hypernym dataset
    hypernyms = np.array(hypernyms)
    import h5py
    f = h5py.File(hypernym_save_path, 'w')
    f.create_dataset('hypernyms', data=hypernyms)
    f.close()


if __name__ == '__main__':

    dataset = 'vg'
    data_config = DatasetConfig(dataset)

    if dataset == 'vrd':
        from open_relation.dataset.vrd.label_hier.obj_hier import objnet
    else:
        from open_relation.dataset.vg.label_hier.obj_hier import objnet

    raw2wn = objnet.raw2wn()

    label2index = objnet.label2index()

    hypernym_save_path = os.path.join(global_config.project_root,
                                      'open_relation', 'label_embedding', 'object',
                                      dataset+'_dataset', 'wordnet_with_'+dataset+'.h5')
    generate_direct_hypernyms(raw2wn, label2index, hypernym_save_path)
