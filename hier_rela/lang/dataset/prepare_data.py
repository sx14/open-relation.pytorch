import json
import os

import numpy as np

from global_config import VRD_ROOT, VG_ROOT
from hier_rela.lang.lang_config import data_config, DATASET_ROOT

'''
This file is used to prepare data for training Hierarchical language model.
It's production is {train | test}_raw_rlts_{dataset}.npy
'''


def collect_raw_rlts(anno_root, id_list_path, rlt_save_path):
    '''
    collect raw relations from annotation files. Here raw means not hierarchical.
    '''
    # load id list
    with open(id_list_path, 'r') as id_list_file:
        anno_list = id_list_file.read().splitlines()

    # raw relationship tuple container
    raw_rlts = []

    # collect raw
    anno_num = len(anno_list)
    for i in range(anno_num):
        print('processsing [%d/%d]' % (anno_num, i + 1))
        anno_path = os.path.join(anno_root, anno_list[i] + '.json')
        anno = json.load(open(anno_path, 'r'))
        anno_rlts = anno['relationships']
        for rlt in anno_rlts:
            anno_obj = rlt['object']
            anno_sbj = rlt['subject']
            anno_pre = rlt['predicate']
            obj_ind = objnet.get_node_by_name(anno_obj['name']).index()
            sbj_ind = objnet.get_node_by_name(anno_sbj['name']).index()
            pre_ind = prenet.get_node_by_name(anno_pre['name']).index()
            raw_rlts.append([sbj_ind, pre_ind, obj_ind, pre_ind])

    raw_rlts = np.array(raw_rlts)
    np.save(rlt_save_path, raw_rlts)
    return raw_rlts


if __name__ == '__main__':
    dataset = 'vrd'

    if dataset == 'vrd':
        dataset_root = VRD_ROOT
        from lib.datasets.vrd.label_hier.obj_hier import objnet
        from lib.datasets.vrd.label_hier.pre_hier import prenet
    else:
        dataset_root = VG_ROOT
        from lib.datasets.vg200.label_hier.obj_hier import objnet
        from lib.datasets.vg200.label_hier.pre_hier import prenet

    anno_root = os.path.join(dataset_root, 'anno')
    split = ('train', 'test')
    for d in split:
        # filename of pictures for training or testing
        list_path = os.path.join(dataset_root, 'ImageSets', 'Main', d + '.txt')
        rlt_save_path = os.path.join(DATASET_ROOT, data_config[d]['raw_rlt_path'] + dataset)
        raw_rlts = collect_raw_rlts(anno_root, list_path, rlt_save_path)
        print('raw relationship tuple num: %d' % len(raw_rlts))
