import scipy.io
from lib.datasets.vrd.label_hier.pre_hier import prenet
import numpy as np

def compute_iou_each(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    if xB < xA or yB < yA:
        IoU = 0
    else:
        area_I = (xB - xA) * (yB - yA)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        IoU = area_I / float(area1 + area2 - area_I)
    return IoU

pred = scipy.io.loadmat('predicate_det_result.mat')
gt = scipy.io.loadmat('gt.mat')


pred_tuple_labels = pred['rlp_labels_ours'][0]
pred_obj_boxes = pred['obj_bboxes_ours'][0]
pred_sbj_boxes = pred['sub_bboxes_ours'][0]

gt_tuple_labels = gt['gt_tuple_label'][0]
gt_obj_boxes = gt['gt_obj_bboxes'][0]
gt_sbj_boxes = gt['gt_sub_bboxes'][0]


N_all = 0
N_score = 0
N_right = 0.0

raw_labels = prenet.get_raw_labels()

for i in range(1000):
    img_pred_tuples = pred_tuple_labels[i]
    img_gt_tuples = gt_tuple_labels[i]
    N_all += len(img_gt_tuples)

    for j in range(len(img_gt_tuples)):
        gt_sbj_box = gt_sbj_boxes[i][j].astype(np.int)
        gt_obj_box = gt_obj_boxes[i][j].astype(np.int)
        gt_sbj = gt_tuple_labels[i][j][0]
        gt_pre = gt_tuple_labels[i][j][1]
        gt_obj = gt_tuple_labels[i][j][2]

        hit = 0.0
        max_scr = 0.0
        for k in range(len(img_pred_tuples)):
            pred_sbj_box = pred_sbj_boxes[i][k].astype(np.int)
            pred_obj_box = pred_obj_boxes[i][k].astype(np.int)
            pred_sbj = pred_tuple_labels[i][k][0]
            pred_pre = pred_tuple_labels[i][k][1]
            pred_obj = pred_tuple_labels[i][k][2]


            ss_iou = compute_iou_each(gt_sbj_box, pred_sbj_box)
            oo_iou = compute_iou_each(gt_sbj_box, pred_sbj_box)

            if ss_iou >= 0.5 and oo_iou >= 0.5 and pred_sbj == gt_sbj and pred_obj == gt_obj:
                gt_label = raw_labels[gt_pre]
                pred_label = raw_labels[pred_pre]
                gt_node = prenet.get_node_by_name(gt_label)
                pred_node = prenet.get_node_by_name(pred_label)
                scr = gt_node.score(pred_node.index())

                if pred_pre == gt_pre:
                    hit = max([hit, 1])

                max_scr = max([max_scr, scr])



        N_right += hit
        N_score += max_scr



print('%.4f' % (N_right / N_all))

print('%.4f' % (N_score / N_all))
