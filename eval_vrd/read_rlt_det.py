import os
import pickle
import scipy.io
import numpy as np

from lib.datasets.vrd.label_hier.pre_hier import prenet
from lib.datasets.vrd.label_hier.obj_hier import objnet
from global_config import PROJECT_ROOT

pred = scipy.io.loadmat('relationship_det_result.mat')
pred_rlt_labels = pred['rlp_labels_ours'][0]
pred_rlt_confs = pred['rlp_confs_ours'][0]
pred_obj_boxes = pred['obj_bboxes_ours'][0]
pred_sbj_boxes = pred['sub_bboxes_ours'][0]


img_paths_mat = scipy.io.loadmat('imagePath.mat')
img_paths = img_paths_mat['imagePath'][0]

pred_roidb = {}


raw_obj_labels = objnet.get_raw_labels()
raw_pre_labels = prenet.get_raw_labels()
for i in range(1000):
    img_path = img_paths[i][0]
    img_id = img_path.split('.')[0]

    sbj_boxes = pred_sbj_boxes[i].astype(np.int)
    obj_boxes = pred_obj_boxes[i].astype(np.int)

    if sbj_boxes.shape[0] == 0:
        continue

    pre_boxes = np.concatenate((
        np.min(np.concatenate((sbj_boxes[:, 0:1], obj_boxes[:, 0:1]), axis=1), axis=1, keepdims=True),
        np.min(np.concatenate((sbj_boxes[:, 1:2], obj_boxes[:, 1:2]), axis=1), axis=1, keepdims=True),
        np.max(np.concatenate((sbj_boxes[:, 2:3], obj_boxes[:, 2:3]), axis=1), axis=1, keepdims=True),
        np.max(np.concatenate((sbj_boxes[:, 3:4], obj_boxes[:, 3:4]), axis=1), axis=1, keepdims=True)), axis=1)

    pred_sbj = pred_rlt_labels[i][:, 0:1]
    pred_pre = pred_rlt_labels[i][:, 1:2]
    pred_obj = pred_rlt_labels[i][:, 2:3]

    for j in range(pred_sbj.shape[0]):
        sbj = pred_sbj[j, 0]
        sbj_label = raw_obj_labels[sbj]
        sbj_node = objnet.get_node_by_name(sbj_label)
        sbj_cls = sbj_node.index()
        pred_sbj[j, 0] = sbj_cls

        obj = pred_obj[j, 0]
        obj_label = raw_obj_labels[obj]
        obj_node = objnet.get_node_by_name(obj_label)
        obj_cls = obj_node.index()
        pred_obj[j, 0] = obj_cls

        pre = pred_pre[j, 0]
        pre_label = raw_pre_labels[pre]
        pre_node = prenet.get_node_by_name(pre_label)
        pre_cls = pre_node.index()
        pred_pre[j, 0] = pre_cls


    pred_confs = pred_rlt_confs[i]

    pred_rois = np.concatenate((pre_boxes, pred_pre,
                                sbj_boxes, pred_sbj,
                                obj_boxes, pred_obj,
                                pred_confs), axis=1)
    pred_roidb[img_id] = pred_rois

save_path = os.path.join(PROJECT_ROOT, 'hier_rela', 'pre_box_label_vrd1.bin')
with open(save_path, 'wb') as f:
    pickle.dump(pred_roidb, f)