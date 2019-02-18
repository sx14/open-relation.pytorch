import pickle
import h5py
import numpy as np
from open_relation.dataset.dataset_config import vrd_predicate_config
from open_relation.dataset.vrd.label_hier.pre_hier import prenet
# from open_relation1.dataset.vrd.predicate.pre_hier import PreNet


dataset_name = 'vrd'
target = 'object'


def eval2(label_vecs, index2label, label2index, pnet):


    labels = pnet.get_all_labels()
    for label in labels:
        label_index = label2index[label]
        label_vec = label_vecs[label_index]
        sub = label_vecs - label_vec


        sub_zero = np.stack((sub, np.zeros(sub.shape)), axis=2)
        relu = np.max(sub_zero, axis=2)
        relu = relu * relu
        E = np.sum(relu, axis=1)
        pred = np.argsort(E)[:20]
        print('\n===== '+label+' =====')
        print('---answer---')
        pre = pnet.get_node_by_name(label)
        wn_label_set = set()
        for hyper_path in pre.hyper_paths():
            for w in hyper_path:
                if w.name() not in wn_label_set:
                    print(w.name())
                    wn_label_set.add(w.name())
        print('---prediction---')
        for p in pred:
            print(index2label[p]+'| %f' % E[p])


def eval3(label_vecs, index2label, label2index, pnet, label):
    label_vec = label_vecs[label2index[label]]
    sub = label_vecs - label_vec
    sub[sub < 0] = 0
    sub_square = sub * sub
    E = np.sum(sub_square, axis=1)
    pred = np.argsort(E)[:20]
    print('\n===== '+label+' =====')
    print('---answer---')

    pre = pnet.get_node_by_name(label)
    wn_label_set = set()
    for hyper_path in pre.hyper_paths():
        for w in hyper_path:
            if w.name() not in wn_label_set:
                print(w.name())
                wn_label_set.add(w.name())
    print('---prediction---')
    for p in pred:
        print(index2label[p] + '| %f' % E[p])


if __name__ == '__main__':
    # label vectors
    weight_path = vrd_predicate_config['label_vec_path']
    label_vec_file = h5py.File(weight_path, 'r')
    label_vecs = np.array(label_vec_file['label_vec'])

    # label mapping
    label2index_path = vrd_predicate_config['label2index_path']
    label2index = pickle.load(open(label2index_path, 'rb'))
    index2label_path = vrd_predicate_config['index2label_path']
    index2label = pickle.load(open(index2label_path, 'rb'))
    raw2path_path = vrd_predicate_config['raw2path_path']
    raw2path = pickle.load(open(raw2path_path, 'rb'))
    # prenet = PreNet()

    eval2(label_vecs, index2label, label2index, prenet)
    # eval3(label_vecs, index2label, label2index, pn, 'sleep next to')
