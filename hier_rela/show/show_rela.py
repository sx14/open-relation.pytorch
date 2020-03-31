import os
import pickle

import cv2
from matplotlib import pyplot as plt

from hier_rela.eval.eval_utils import *
from global_config import VRD_ROOT, VG_ROOT
from hier_det.tools.show_box import show_boxes, draw_boxes
from lib.model.nms.nms_cpu import nms_cpu as py_cpu_nms

# vrd - vg
dataset = 'vg'
# rela - pre
target = 'rela'


def show_img_relas(gt_roidb, pred_roidb, img, objnet, prenet, thr):
    keep = pred_roidb[:, -2] > 0
    hit_roidb = pred_roidb[keep, :]
    sbj_dets = hit_roidb[:, 5:10]
    obj_dets = hit_roidb[:, 10:15]

    dets = np.concatenate((sbj_dets, obj_dets), axis=0)

    uni_dets = np.unique(dets, axis=0)
    keep = py_cpu_nms(uni_dets, thr)
    uni_dets = uni_dets[keep]

    uni_det_boxes = uni_dets[:, :4]
    uni_det_confs = np.zeros(uni_dets.shape[0])
    uni_det_labels = []
    for i in range(uni_dets.shape[0]):
        uni_det_cls = uni_dets[i, 4]
        label = objnet.get_node_by_index(int(uni_det_cls)).name()
        uni_det_labels.append(label)

    gt_print = np.ones(gt_roidb.shape[0])
    for i in range(pred_roidb.shape[0]):
        if pred_roidb[i, -5] > -1:  # box hit
            pre_cls = pred_roidb[i, 4]
            sbj_cls = pred_roidb[i, 9]
            obj_cls = pred_roidb[i, 14]

            pre_label = prenet.get_node_by_index(int(pre_cls)).name().split('.')[0]
            sbj_label = objnet.get_node_by_index(int(sbj_cls)).name().split('.')[0]
            obj_label = objnet.get_node_by_index(int(obj_cls)).name().split('.')[0]

            pre_gt = pred_roidb[i, -4]
            sbj_gt = pred_roidb[i, -3]
            obj_gt = pred_roidb[i, -2]

            pre_gt_label = prenet.get_node_by_index(int(pre_gt))
            sbj_gt_label = objnet.get_node_by_index(int(sbj_gt))
            obj_gt_label = objnet.get_node_by_index(int(obj_gt))

            if pred_roidb[i, -1] > 0:  # rela hit
                k = pred_roidb[i, -5]
                gt_print[int(k)] = 0

                print('%.2f <%s, %s, %s>|<%s, %s, %s>' % (pred_roidb[i, -2], sbj_gt_label, pre_gt_label, obj_gt_label,
                                                          sbj_label, pre_label, obj_label))
            else:
                print('%.2f             |<%s, %s, %s>' % (pred_roidb[i, -2], sbj_label, pre_label, obj_label))

    for i in range(gt_print.shape[0]):
        if gt_print[i] == 1:
            pre_gt = gt_roidb[i, 4]
            sbj_gt = gt_roidb[i, 9]
            obj_gt = gt_roidb[i, 14]

            pre_gt_label = prenet.get_node_by_index(int(pre_gt))
            sbj_gt_label = objnet.get_node_by_index(int(sbj_gt))
            obj_gt_label = objnet.get_node_by_index(int(obj_gt))

            print('%.2f <%s, %s, %s>|              ' % (0.0, sbj_gt_label, pre_gt_label, obj_gt_label))

    dets_temp = np.copy(uni_det_boxes)
    dets_temp[:, 2] = uni_det_boxes[:, 2] - uni_det_boxes[:, 0]  # width
    dets_temp[:, 3] = uni_det_boxes[:, 3] - uni_det_boxes[:, 1]  # height
    show_boxes(img, dets_temp, uni_det_labels, uni_det_confs, 'all')

    # --- save ----
    img_bgr = np.stack((img[:, :, 2], img[:, :, 1], img[:, :, 0]), axis=2)
    im_sav = draw_boxes(img_bgr, dets_temp)
    cv2.imwrite('temp.jpg', im_sav)


if dataset == 'vrd':
    ds_root = VRD_ROOT
    from lib.datasets.vrd.label_hier.obj_hier import objnet
    from lib.datasets.vrd.label_hier.pre_hier import prenet
else:
    ds_root = VG_ROOT
    from lib.datasets.vglsj.label_hier.obj_hier import objnet
    from lib.datasets.vglsj.label_hier.pre_hier import prenet

if target == 'pre':
    box_thr = 1.0
else:
    box_thr = 0.5

gt_roidb_path = '../gt_rela_roidb_%s.bin' % dataset
gt_roidb = pickle.load(open(gt_roidb_path))

pred_roidb_path = '../%s_box_label_%s_hier.bin' % (target, dataset)
pred_roidb = pickle.load(open(pred_roidb_path))

img_root = os.path.join(ds_root, 'JPEGImages')

for img_id in gt_roidb:
    curr_gt = gt_roidb[img_id]
    curr_gt = np.array(curr_gt)
    if img_id not in pred_roidb:
        continue

    curr_pr = pred_roidb[img_id]
    img_path = os.path.join(img_root, img_id + '.jpg')
    im = plt.imread(img_path)
    if im is None or curr_gt is None or curr_pr is None or curr_gt.shape[0] == 0 or curr_pr.shape[0] == 0:
        continue

    show_img_relas(curr_gt, curr_pr, im, objnet, prenet, 0.3)
