"""
step1 split original annotation json package
next: wash_anno.py
"""
import os
import cv2
import h5py
import json
import numpy as np


def nms_cpu(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def nms(boxes, labels):
    uni_labels = np.unique(labels)

    nms_boxes = []
    nms_labels = []

    # for each class
    for uni_label in uni_labels:
        same_cls_inds = np.where(labels == uni_label)[0]
        same_cls_boxes = boxes[same_cls_inds, :]

        fake_scores = np.ones((same_cls_boxes.shape[0], 1))
        fake_dets = np.concatenate((same_cls_boxes, fake_scores), axis=1)

        keep = nms_cpu(fake_dets, 0.6)
        same_cls_boxes = same_cls_boxes[keep].tolist()
        same_labels = np.zeros((same_cls_boxes.shape[0])).astype(np.int)
        same_labels[:] = uni_label
        same_labels = same_labels.tolist()

        nms_boxes += same_cls_boxes
        nms_labels += same_labels

    return nms_boxes, nms_labels







def split_vts_pkg(vts_anno_path, dataset_root, img_root, json_root, img_list_root):
    if not os.path.exists(vts_anno_path):
        print(vts_anno_path+' not exists.')
        exit(-1)

    vts_anno_pkg = h5py.File(vts_anno_path, 'r')
    vts_meta = vts_anno_pkg['meta']
    vts_obj_cls = vts_meta['cls']['idx2name']
    vts_pre_cls = vts_meta['pre']['idx2name']

    obj_cls = []
    pre_cls = []

    for i in range(1, 201):     # 0 is background
        obj = np.array(vts_obj_cls[str(i)]).tolist()
        obj_cls.append(obj)
    for i in range(0, 100):     # no background
        pre = np.array(vts_pre_cls[str(i)]).tolist()
        pre_cls.append(pre)

    vts_gt = vts_anno_pkg['gt']

    split_names = ['train', 'test']

    for split in split_names:
        vts_split = vts_gt[split]
        img_ids = np.array(vts_split).tolist()
        N_img = len(img_ids)

        # get image json anno
        for i, img_id in enumerate(img_ids):
            print('process [%d/%d]' % (N_img, i+1))

            # part1: image_info
            img_path = os.path.join(img_root, img_id+'.jpg')
            if not os.path.exists(img_path):
                print(img_path+' not exists.')
                continue

            img = cv2.imread(img_path)
            img_h = img.shape[0]
            img_w = img.shape[1]
            img_info = {
                'image_id': img_id,
                'height': img_h,
                'width': img_w,
            }

            # raw anno
            vts_anno = vts_split[img_id]
            vts_obj_boxes = np.array(vts_anno['obj_boxes'])
            vts_sbj_boxes = np.array(vts_anno['sub_boxes'])
            vts_rlt_triplets = np.array(vts_anno['rlp_labels'])

            vts_all_obj_boxes = np.concatenate((vts_sbj_boxes, vts_obj_boxes), 0)
            vts_all_obj_labels = np.concatenate((vts_rlt_triplets[:, 0], vts_rlt_triplets[:, 2]))

            # remove redundant
            vts_all_obj_boxes, inds = np.unique(vts_all_obj_boxes, return_index=True, axis=0)
            vts_all_obj_labels = vts_all_obj_labels[inds]
            vts_all_obj_boxes, vts_all_obj_labels = nms(vts_all_obj_boxes, vts_all_obj_labels)


            # part2: objects
            objs = []
            for o, vts_obj_box in enumerate(vts_all_obj_boxes):
                obj = {
                    'xmin': int(vts_obj_box[0]),
                    'ymin': int(vts_obj_box[1]),
                    'xmax': int(vts_obj_box[2]),
                    'ymax': int(vts_obj_box[3]),
                    'name': obj_cls[vts_all_obj_labels[o]-1],
                    'synsets': ['entity.n.01']
                }
                objs.append(obj)

            # part3: relationships
            rlts = []
            for r, vts_rlt_triplet in enumerate(vts_rlt_triplets):
                vts_sbj_box = vts_sbj_boxes[r]
                vts_obj_box = vts_obj_boxes[r]
                rlt = {
                    'subject': {
                        'xmin': int(vts_sbj_box[0]),
                        'ymin': int(vts_sbj_box[1]),
                        'xmax': int(vts_sbj_box[2]),
                        'ymax': int(vts_sbj_box[3]),
                        'name': obj_cls[vts_rlt_triplet[0]-1],
                        'synsets': ['entity.n.01']
                    },

                    'object': {
                        'xmin': int(vts_obj_box[0]),
                        'ymin': int(vts_obj_box[1]),
                        'xmax': int(vts_obj_box[2]),
                        'ymax': int(vts_obj_box[3]),
                        'name': obj_cls[vts_rlt_triplet[2]-1],
                        'synsets': ['entity.n.01']
                    },

                    'predicate': {
                        'xmin': int(min(vts_sbj_box[0], vts_obj_box[0])),
                        'ymin': int(min(vts_sbj_box[1], vts_obj_box[1])),
                        'xmax': int(max(vts_sbj_box[2], vts_obj_box[2])),
                        'ymax': int(max(vts_sbj_box[3], vts_obj_box[3])),
                        'name': pre_cls[vts_rlt_triplet[1]],
                        'synsets': ['relation']
                    }
                }
                rlts.append(rlt)

            new_anno = {
                'relationships': rlts,
                'objects': objs,
                'image_info': img_info
            }

            json_path = os.path.join(json_root, img_id + '.json')
            with open(json_path, 'w') as f:
                json.dump(new_anno, f)

        # save split image list
        for l in range(len(img_ids)):
            img_ids[l] = img_ids[l] + '\n'
        img_list_path = os.path.join(img_list_root, split + '.txt')
        with open(img_list_path, 'w') as f:
            f.writelines(img_ids)

    # save class list
    for i in range(len(obj_cls)):
        obj_cls[i] = obj_cls[i] + '\n'
    for i in range(len(pre_cls)):
        pre_cls[i] = pre_cls[i] + '\n'

    obj_cls_path = os.path.join(dataset_root, 'object_labels.txt')
    pre_cls_path = os.path.join(dataset_root, 'predicate_labels.txt')

    with open(obj_cls_path, 'w') as f:
        f.writelines(obj_cls)
    with open(pre_cls_path, 'w') as f:
        f.writelines(pre_cls)

    vts_anno_pkg.close()


def split_anno_pkg(vg_config):
    vts_anno_path = vg_config['vts_anno_path']
    json_root = vg_config['clean_anno_root']
    img_root = vg_config['JPEGImages']
    img_list_root = vg_config['ImageSets']
    ds_root = vg_config['ds_root']

    if os.path.exists(json_root):
        import shutil
        shutil.rmtree(json_root)
        os.makedirs(json_root)

    split_vts_pkg(vts_anno_path, ds_root, img_root, json_root, img_list_root)


