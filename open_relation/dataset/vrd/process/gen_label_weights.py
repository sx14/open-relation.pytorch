"""
weights for original labels only
"""

import os
import pickle
import numpy as np
from open_relation.dataset.dataset_config import DatasetConfig
from open_relation.dataset.vrd.label_hier.pre_hier import prenet
from open_relation.dataset.vrd.label_hier.obj_hier import objnet


def gen_weights(box_labels, vrd2path, index2label, label2index, weights_save_path, mode):
    # WARNING: deprecated
    # counter
    label_counter = np.zeros(len(index2label))
    vrd_counter = np.zeros(len(index2label))

    # counting
    for img_id in box_labels:
        img_box_labels = box_labels[img_id]
        for box_label in img_box_labels:
            vrd_label = box_label[4]
            vrd_counter[label2index[vrd_label]] += 1
            label_path = vrd2path[label2index[vrd_label]]
            for l in label_path:
                label_counter[l] += 1

    if mode == 'org':
        show_counter = vrd_counter
    else:
        show_counter = label_counter

    ranked_counts = np.sort(show_counter).tolist()
    ranked_counts.reverse() # large -> small
    ranked_counts = np.array(ranked_counts)
    ranked_counts = ranked_counts[ranked_counts > 0]

    ranked_inds = np.argsort(show_counter).tolist()
    ranked_inds.reverse()   # large -> small
    ranked_inds = ranked_inds[:len(ranked_counts)]

    count_sum = ranked_counts.sum()
    min_weight = 1.0 / (ranked_counts.max() / count_sum)
    vrd2weight = dict()
    for i in range(len(ranked_counts)):
        w = 1.0 / (ranked_counts[i] / count_sum) / min_weight
        vrd2weight[ranked_inds[i]] = w
    pickle.dump(vrd2weight, open(weights_save_path, 'wb'))


def gen_weights1(box_labels, vrd2path, index2label, label2index, weights_save_path, mode):
    # counter
    label_counter = np.zeros(len(index2label))
    vrd_counter = np.zeros(len(index2label))

    # counting
    for img_id in box_labels:
        img_box_labels = box_labels[img_id]
        for box_label in img_box_labels:
            vrd_label = box_label[4]
            vrd_counter[label2index[vrd_label]] += 1
            label_path = vrd2path[label2index[vrd_label]]
            for l in label_path:
                label_counter[l] += 1

    if mode == 'raw':
        show_counter = vrd_counter
    else:
        show_counter = label_counter

    ranked_counts = np.sort(show_counter).tolist()
    ranked_counts.reverse() # large -> small
    ranked_counts = np.array(ranked_counts)
    ranked_counts = ranked_counts[ranked_counts > 0]

    ranked_inds = np.argsort(show_counter).tolist()
    ranked_inds.reverse()   # large -> small
    ranked_inds = ranked_inds[:len(ranked_counts)]

    count_sum = ranked_counts.sum()
    max_weight = 5.0
    min_weight = 1.0
    k = (max_weight - min_weight) / (ranked_counts[-1] - ranked_counts[0])
    b = max_weight - k * ranked_counts[-1]
    vrd2weight = dict()
    # expected max weight: 10
    for i in range(len(ranked_counts)):
        w = k * ranked_counts[i] + b
        vrd2weight[ranked_inds[i]] = w
    pickle.dump(vrd2weight, open(weights_save_path, 'wb'))


def gen_label_weigths(target):
    # label maps
    if target == 'object':
        labelnet = objnet
    elif target == 'predicate':
        labelnet = prenet
    else:
        print('Target is wrong!')
        exit(-1)

    raw2path = labelnet.raw2path()
    index2label = labelnet.index2label()
    label2index = labelnet.label2index()

    # org data
    dataset_config = DatasetConfig('vrd')
    prepare_root = dataset_config.extra_config[target].prepare_root
    box_label_path = os.path.join(prepare_root, 'train_box_label.bin')
    box_labels = pickle.load(open(box_label_path, 'rb'))

    # weight save path
    weights_save_path = dataset_config.extra_config[target].config['raw2weight_path']
    gen_weights1(box_labels, raw2path, index2label, label2index, weights_save_path, 'raw')
