import pickle
import h5py
import numpy as np
from nltk.corpus import wordnet as wn
from open_relation.dataset.dataset_config import DatasetConfig


dataset_name = 'vrd'
target = 'object'

data_config = DatasetConfig(dataset_name)

if dataset_name == 'vrd' and target == 'object':
    from open_relation.dataset.vrd.label_hier.obj_hier import objnet as classnet
elif dataset_name == 'vrd' and target == 'predicate':
    from open_relation.dataset.vrd.label_hier.pre_hier import prenet as classnet
elif dataset_name == 'vg' and target == 'object':
    from open_relation.dataset.vg.label_hier.obj_hier import objnet as classnet
else:
    from open_relation.dataset.vg.label_hier.pre_hier import prenet as classnet


def eval2(label_vecs, labelnet):
    label2index = labelnet.label2index()
    index2label = labelnet.index2label()
    vg_labels = labelnet.get_raw_labels()
    for vg_label in vg_labels:
        vg_label_index = label2index[vg_label]
        vg_label_vec = label_vecs[vg_label_index]
        sub = label_vecs - vg_label_vec

        # wn_label_index = label2index[vg2wn[vg_label][0]]
        # wn_label_vec = label_vecs[wn_label_index]
        # sub = label_vecs - wn_label_vec

        sub_zero = np.stack((sub, np.zeros(sub.shape)), axis=2)
        relu = np.max(sub_zero, axis=2)
        relu = relu * relu
        E = np.sum(relu, axis=1)
        pred = np.argsort(E)[:20]
        print('\n===== '+vg_label+' =====')
        print('---answer---')
        gt_node = labelnet.get_node_by_name(vg_label)
        for hyper_path in gt_node.hyper_paths():
            for w in hyper_path:
                print(w.name())
        print('---prediction---')
        for p in pred:
            print(index2label[p]+'| %f' % E[p])





def eval4(label_vecs, label2index, label1, label2):
    ind1 = label2index[label1]
    ind2=  label2index[label2]
    vec1 = label_vecs[ind1]
    vec2 = label_vecs[ind2]
    sub = vec1 - vec2
    sub[sub < 0] = 0
    sub_square = sub * sub
    E = np.sqrt(np.sum(sub_square))
    print('P( %s , %s ) = %.2f' % (label1, label2, E))



if __name__ == '__main__':
    # label vectors
    weight_path = data_config.extra_config[target].config['label_vec_path']
    label_vec_file = h5py.File(weight_path, 'r')
    label_vecs = np.array(label_vec_file['label_vec'])

    # label mapping
    label2index = classnet.label2index()
    index2label = classnet.index2label()

    eval2(label_vecs, classnet)
    # eval4(label_vecs, label2index, 'shirt', 'garment.n.01')
    # objnet.get_node_by_name('person').show_hyper_paths()
