# coding=utf-8

import os
import numpy as np
import cv2


def compute_iou(box, proposal):
    """
    compute the IoU between box with proposal
    Arg:
        box: [x1,y1,x2,y2]
        proposal: N*4 matrix, each line is [p_x1,p_y1,p_x2,p_y2]
    output:
        IoU: N*1 matrix, every IoU[i] means the IoU between
             box with proposal[i,:]
    """
    len_proposal = np.shape(proposal)[0]
    IoU = np.empty([len_proposal, 1])
    for i in range(len_proposal):
        xA = max(box[0], proposal[i, 0])
        yA = max(box[1], proposal[i, 1])
        xB = min(box[2], proposal[i, 2])
        yB = min(box[3], proposal[i, 3])

        if xB < xA or yB < yA:
            IoU[i, 0] = 0
        else:
            area_I = (xB - xA) * (yB - yA)
            area1 = (box[2] - box[0]) * (box[3] - box[1])
            area2 = (proposal[i, 2] - proposal[i, 0]) * (proposal[i, 3] - proposal[i, 1])
            IoU[i, 0] = area_I / float(area1 + area2 - area_I)
    return IoU


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


def rela_recall(gt_roidb, pred_roidb, N_recall):
    N_right = 0.0
    N_total = 0.0
    N_data = len(gt_roidb.keys())
    N_box_goot = 0.0
    N_det_goot = 0.0
    num_right = {}
    for image_id in gt_roidb:
        num_right[image_id] = 0
        # px1, py1, px2, py2, pname, sx1, sy1, sx2, sy2, sname, ox1, oy1, ox2, oy2, oname
        curr_gt_roidb = np.array(gt_roidb[image_id])

        if len(curr_gt_roidb) == 0:
            continue
        rela_gt = curr_gt_roidb[:, 4]
        sub_box_gt = curr_gt_roidb[:, 5:10]
        obj_box_gt = curr_gt_roidb[:, 10:15]
        sub_gt = curr_gt_roidb[:, 9]
        obj_gt = curr_gt_roidb[:, 14]
        box_gt = np.concatenate((sub_box_gt, obj_box_gt))
        box_gt = np.unique(box_gt, axis=0)

        N_rela = len(rela_gt)
        N_total = N_total + N_rela

        # px1, py1, px2, py2, pname, sx1, sy1, sx2, sy2, sname, ox1, oy1, ox2, oy2, oname, rlt_score
        curr_pred_roidb = np.array(pred_roidb[image_id])
        if len(curr_pred_roidb) == 0:
            continue
        rela_pred = curr_pred_roidb[:, 4]
        rela_pred_score = curr_pred_roidb[:, -1]
        sub_box_dete = curr_pred_roidb[:, 5:10]
        obj_box_dete = curr_pred_roidb[:, 10:15]
        sub_dete = curr_pred_roidb[:, 9]
        obj_dete = curr_pred_roidb[:, 14]
        box_det = np.unique(sub_box_dete, axis=0)

        N_pred = len(rela_pred)

        for b in range(box_gt.shape[0]):
            gt = box_gt[b]
            box_hit = 0
            det_hit = 0
            for o in range(box_det.shape[0]):
                det = box_det[o]
                if compute_iou_each(gt, det) > 0.5:
                    box_hit = 1
                    if gt[4] == det[4]:
                        det_hit = 1

            N_box_goot += box_hit
            N_det_goot += det_hit



        sort_score = np.sort(rela_pred_score)[::-1]
        if N_recall >= N_pred:
            thresh = float('-Inf')
        else:
            thresh = sort_score[N_recall]

        detected_gt = np.zeros([N_rela, ])
        for j in range(N_pred):
            if rela_pred_score[j] <= thresh:
                continue

            for k in range(N_rela):
                if detected_gt[k] == 1:
                    continue
                if (sub_gt[k] == sub_dete[j]) and (obj_gt[k] == obj_dete[j]) and (rela_gt[k] == rela_pred[j]):
                    s_iou = compute_iou_each(sub_box_dete[j], sub_box_gt[k])
                    o_iou = compute_iou_each(obj_box_dete[j], obj_box_gt[k])
                    if (s_iou >= 0.5) and (o_iou >= 0.5):
                        detected_gt[k] = 1
                        N_right = N_right + 1
                        num_right[image_id] = num_right[image_id] + 1

    print('Proposal recall: %.4f' % (N_box_goot / N_total))
    print('Detection recall: %.4f' % (N_det_goot / N_total))
    acc = N_right / N_total
    return acc, num_right
