import os
import pickle
import numpy as np
from open_relation.dataset.vrd.label_hier.pre_hier import prenet
# from open_relation1.dataset.vrd.predicate.pre_hier import PreNet
from open_relation.dataset.dataset_config import vrd_predicate_config
from open_relation import global_config


def generate_direct_hypernyms(pre_net, label2index, hypernym_save_path):
    # ==== generate direct hypernym relations ====
    # [[hypo, hyper]]
    hypernyms = []
    for label in pre_net.get_all_labels():
        pre = pre_net.get_node_by_name(label)
        hypers = pre.hypers()
        for hyper in hypers:
            hypernyms.append([label2index[label], label2index[hyper.name()]])

    # save hypernym dataset
    hypernyms = np.array(hypernyms)
    import h5py
    f = h5py.File(hypernym_save_path, 'w')
    f.create_dataset('hypernyms', data=hypernyms)
    f.close()


if __name__ == '__main__':

    # prenet = PreNet()
    label2index_path = vrd_predicate_config['label2index_path']
    label2index = pickle.load(open(label2index_path, 'rb'))

    hypernym_save_path = os.path.join(global_config.project_root,
                                      'open_relation1', 'label_embedding', 'predicate', 'vrd_dataset', 'wordnet_with_vrd.h5')
    generate_direct_hypernyms(prenet , label2index, hypernym_save_path)