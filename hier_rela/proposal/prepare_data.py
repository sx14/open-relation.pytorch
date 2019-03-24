import os
import numpy as np
import json
import random
from lang_config import data_config
from lib.datasets.vrd.label_hier.obj_hier import objnet
from lib.datasets.vrd.label_hier.pre_hier import prenet
from global_config import VRD_ROOT, VG_ROOT


def extend_neg_samples(im_boxes):
    pos_samples = im_boxes
    neg_samples = pos_samples.copy()

    sbj_boxes = pos_samples[:, 5:10]
    obj_boxes = pos_samples[:, 10:15]

    if pos_samples.shape[0] < 3:
        # exchange sbj and obj
        neg_samples[:, 5:10] = pos_samples[:, 10:15]
        neg_samples[:, 10:15] = pos_samples[:, 5:10]
    else:
        # gen random pairs
        all_boxes = np.concatenate((sbj_boxes, obj_boxes), 0)
        all_boxes = np.unique(all_boxes, axis=0)

        if all_boxes.shape[0] < neg_samples.shape[0]:
            neg_samples = neg_samples[:all_boxes.shape[0]]
            pos_samples = pos_samples[:all_boxes.shape[0]]

        rand_obj_inds = random.sample(range(all_boxes.shape[0]), neg_samples.shape[0])
        rand_sbj_inds = random.sample(range(all_boxes.shape[0]), neg_samples.shape[0])
        rand_objs = all_boxes[rand_obj_inds]
        rand_sbjs = all_boxes[rand_sbj_inds]

        neg_samples[:, 5:10] = rand_sbjs
        neg_samples[:, 10:15] = rand_objs

        neg_pre_xmins = np.min(np.stack((rand_sbjs[:, 0], rand_objs[:, 0]), 1), axis=1)
        neg_pre_ymins = np.min(np.stack((rand_sbjs[:, 1], rand_objs[:, 1]), 1), axis=1)
        neg_pre_xmaxs = np.max(np.stack((rand_sbjs[:, 2], rand_objs[:, 2]), 1), axis=1)
        neg_pre_ymaxs = np.max(np.stack((rand_sbjs[:, 3], rand_objs[:, 3]), 1), axis=1)

        neg_samples[:, 0] = neg_pre_xmins
        neg_samples[:, 1] = neg_pre_ymins
        neg_samples[:, 2] = neg_pre_xmaxs
        neg_samples[:, 3] = neg_pre_ymaxs

    neg_pre_labels = np.zeros(neg_samples.shape[0])
    neg_samples[:, 4] = neg_pre_labels

    pos_neg_samples = np.concatenate((pos_samples, neg_samples), 0)
    return pos_neg_samples



def collect_raw_rlts(anno_root, id_list_path, rlt_save_path):

    # load id list
    with open(id_list_path, 'r') as id_list_file:
        anno_list = id_list_file.read().splitlines()

    # raw relationship tuple container
    raw_rlts = []

    # collect raw
    anno_num = len(anno_list)
    for i in range(anno_num):
        print('processsing [%d/%d]' % (anno_num, i+1))
        anno_path = os.path.join(anno_root, anno_list[i]+'.json')
        anno = json.load(open(anno_path, 'r'))
        anno_rlts = anno['relationships']
        for rlt in anno_rlts:
            anno_obj = rlt['object']
            anno_sbj = rlt['subject']
            anno_pre = rlt['predicate']
            # print('<%s, %s, %s>' % (anno_sbj['name'], anno_pre['name'], anno_obj['name']))
            obj_ind = objnet.get_node_by_name(anno_obj['name']).index()
            sbj_ind = objnet.get_node_by_name(anno_sbj['name']).index()
            pre_ind = prenet.get_node_by_name(anno_pre['name']).index()

            obj_box = [anno_obj['xmin'], anno_obj['ymin'], anno_obj['xmax'], anno_obj['ymax']]
            sbj_box = [anno_sbj['xmin'], anno_sbj['ymin'], anno_sbj['xmax'], anno_sbj['ymax']]
            pre_box = [
                min([obj_box[0], sbj_box[0]]),
                min([obj_box[1], sbj_box[1]]),
                max([obj_box[2], sbj_box[2]]),
                max([obj_box[3], sbj_box[3]]),
            ]

            pre_box.append(pre_ind)
            sbj_box.append(sbj_ind)
            obj_box.append(obj_ind)

            raw_rlts.append(pre_box + sbj_box + obj_box)

    raw_rlts = np.array(raw_rlts)
    ext_rlts = extend_neg_samples(raw_rlts)
    np.save(rlt_save_path, ext_rlts)
    return raw_rlts


def equal_interval_prob(num):
    interval = 1.0 * 2 / (num * (num + 1))
    probs = [interval * i for i in range(num + 1)][1:]
    return probs


# def extend_rlts(raw_rlts, rlt_save_path):
#     new_rlts = []
#     raw_rlt_num = len(raw_rlts)
#     for i, raw_rlt in enumerate(raw_rlts):
#         print('processing [%d/%d]' % (raw_rlt_num, i+1))
#         pre_ind = raw_rlt[1]
#         pre_node = prenet.get_node_by_index(pre_ind)
#         pre_hyper_inds = pre_node.trans_hyper_inds()
#
#         obj_sample_num = 5
#         # extend hyper predicate
#         for p_ind in pre_hyper_inds:
#             # raw subject, hyper_pre, raw object, raw pre
#             new_rlts.append([raw_rlt[0], p_ind, raw_rlt[2], pre_ind])
#             sbj_hyper_inds = objnet.get_node_by_index(raw_rlt[0]).trans_hyper_inds()
#             sbj_sample_probs = equal_interval_prob(len(sbj_hyper_inds))
#             sbj_samples = np.random.choice(sbj_hyper_inds, obj_sample_num, p=sbj_sample_probs)
#             obj_hyper_inds = objnet.get_node_by_index(raw_rlt[2]).trans_hyper_inds()
#             obj_sample_probs = equal_interval_prob(len(obj_hyper_inds))
#             obj_samples = np.random.choice(obj_hyper_inds, obj_sample_num, p=obj_sample_probs)
#
#             # extend hyper object
#             for i in range(obj_sample_num):
#                 # hyper subject, hyper_pre, hyper object, raw pre
#                 new_rlts.append([sbj_samples[i], p_ind, obj_samples[i], pre_ind])
#
#     new_rlts = np.array(new_rlts)
#     np.save(rlt_save_path, new_rlts)
#     return new_rlts


if __name__ == '__main__':
    dataset = 'vrd'

    if dataset == 'vrd':
        dataset_root = VRD_ROOT
    else:
        dataset_root = VG_ROOT

    anno_root = os.path.join(dataset_root, 'anno')
    split = ['train', 'test']
    for d in split:
        list_path = os.path.join(dataset_root, 'ImageSets', 'Main', d + '.txt')
        rlt_save_path = data_config[d]['raw_rlt_path']+dataset
        raw_rlts = collect_raw_rlts(anno_root, list_path, rlt_save_path)
        print('raw relationship tuple num: %d' % len(raw_rlts))
        # rlt_save_path = data_config[d]['ext_rlt_path']+dataset
        # ext_rlts = extend_rlts(raw_rlts, rlt_save_path)
        # print('extended relationship tuple num: %d' % len(ext_rlts))