import os
import json
import numpy as np
from open_relation.dataset.dataset_config import DatasetConfig


def load_list(list_path):
    if not os.path.exists(list_path):
        print('"%s" not exists.' % list_path)
    with open(list_path, 'r') as f:
        index2label = f.readlines()
        for i in range(len(index2label)):
            index2label[i] = index2label[i].strip()
    return index2label


def rlt_reformat(rlt_anno, obj_ind2label, pre_ind2label):

    def obj_reformat(obj_anno, obj_ind2label):
        obj = dict()
        obj['name'] = obj_ind2label[int(obj_anno['category'])]
        obj['ymin'] = int(obj_anno['bbox'][0])
        obj['ymax'] = int(obj_anno['bbox'][1])
        obj['xmin'] = int(obj_anno['bbox'][2])
        obj['xmax'] = int(obj_anno['bbox'][3])
        return obj

    sbj_anno = rlt_anno['subject']
    obj_anno = rlt_anno['object']
    sbj = obj_reformat(sbj_anno, obj_ind2label)
    obj = obj_reformat(obj_anno, obj_ind2label)
    pre = dict()
    pre['name'] = pre_ind2label[int(rlt_anno['predicate'])]
    # predicate box is union of obj box and sbj box
    pre['ymin'] = min(obj['ymin'], sbj['ymin'])
    pre['ymax'] = max(obj['ymax'], sbj['ymax'])
    pre['xmin'] = min(obj['xmin'], sbj['xmin'])
    pre['xmax'] = max(obj['xmax'], sbj['xmax'])
    new_rlt = dict()
    new_rlt['object'] = obj
    new_rlt['subject'] = sbj
    new_rlt['predicate'] = pre
    return new_rlt



def reformat_anno():
    dataset_config = DatasetConfig('vrd')
    org_anno_root = dataset_config.data_config['dirty_anno_root']
    dst_anno_root = dataset_config.data_config['clean_anno_root']

    # load vrd label list
    obj_label_list_path = os.path.join(dataset_config.dataset_root, 'object_labels.txt')
    obj_ind2label = load_list(obj_label_list_path)

    pre_label_list_path = os.path.join(dataset_config.dataset_root, 'predicate_labels.txt')
    pre_ind2label = load_list(pre_label_list_path)

    # all dirty annotation files
    anno_list = os.listdir(org_anno_root)
    for i, anno_name in enumerate(anno_list):
        print('processing [%d/%d]' % (len(anno_list), i+1))

        org_anno_path = os.path.join(org_anno_root, anno_name)
        org_anno = json.load(open(org_anno_path, 'r'))

        # for removing redundant objects from predicate
        obj_label_boxes = []

        # clean anno collection
        rlts = []
        for rlt in org_anno:
            # convert predicate anno
            new_rlt = rlt_reformat(rlt, obj_ind2label, pre_ind2label)
            rlts.append(new_rlt)

            obj_sbj = [rlt['object'], rlt['subject']]
            for obj in obj_sbj:
                # left top, right bottom
                # ymin, ymax, xmin, xmax, category
                label_box = obj['bbox']
                label_box.append(obj['category'])
                obj_label_boxes.append(label_box)

        objs = []
        # remove redundant objects
        if len(obj_label_boxes) > 0:
            obj_label_boxes = np.array(obj_label_boxes)
            unique_label_boxes = np.unique(obj_label_boxes, axis=0)
            for label_box in unique_label_boxes:
                obj = dict()
                obj['name'] = obj_ind2label[int(label_box[4])].strip()
                obj['ymin'] = int(label_box[0])
                obj['ymax'] = int(label_box[1])
                obj['xmin'] = int(label_box[2])
                obj['xmax'] = int(label_box[3])
                objs.append(obj)

        new_anno = dict()
        new_anno['objects'] = objs
        new_anno['relations'] = rlts
        save_path = os.path.join(dst_anno_root, anno_name)
        json.dump(new_anno, open(save_path, 'w'))
