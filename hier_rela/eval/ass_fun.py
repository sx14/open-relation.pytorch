# coding=utf-8

import os
import numpy as np
import cv2


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


def rela_recall(mode, gt_roidb, pred_roidb, N_recall, objnet, prenet, box_thr=0.5, alpha=1):
    N_rela_right = 0.0
    N_pre_right = 0.0
    N_rela_total = 0.0

    N_obj_total = 0.0
    N_obj_box_good = 0.0
    N_obj_det_good = 0.0

    N_zero_rela_right = 0.0
    N_zero_pre_right = 0.0
    N_zero_rela_total = 0.0

    N_img = len(gt_roidb.keys())
    results = {}
    gt_eval = {}

    for i, image_id in enumerate(gt_roidb):
        print('pred [%d/%d] %s' % (N_img, i + 1, image_id))
        results[image_id] = {
            'N_rlt': 0,
            'N_rlt_right': 0,
            'N_rlt_box_right': 0,
            'N_rlt_pair_right': 0,

            'N_rlt_gt': 0,
            'N_rlt_gt_right': 0,
            'N_rlt_box_gt_right': 0,
            'N_rlt_pair_gt_right': 0,

            'N_obj': 0,
            'N_obj_box_right': 0,
            'N_obj_det_right': 0,

            'N_obj_gt': 0,
            'N_obj_box_gt_right': 0,
            'N_obj_det_gt_right': 0,
        }

        # px1, py1, px2, py2, pid, sx1, sy1, sx2, sy2, sid, ox1, oy1, ox2, oy2, oid, pconf, sconf, oconf, zero, count
        curr_gt_roidb = np.array(gt_roidb[image_id])
        # pid, sid, oid, score
        curr_gt_eval = np.zeros((curr_gt_roidb.shape[0], 4))
        gt_eval[image_id] = curr_gt_eval

        if len(curr_gt_roidb) == 0:
            continue

        sub_box_gt = curr_gt_roidb[:, 5:10]
        obj_box_gt = curr_gt_roidb[:, 10:15]
        rela_gt = curr_gt_roidb[:, 4].astype(np.int).tolist()
        sub_gt = curr_gt_roidb[:, 9].astype(np.int).tolist()
        obj_gt = curr_gt_roidb[:, 14].astype(np.int).tolist()
        box_gt = np.concatenate((sub_box_gt, obj_box_gt))
        box_gt = np.unique(box_gt, axis=0)
        count_gt = curr_gt_roidb[:, 19]

        if alpha == 1:
            count_gt[:] = 1

        N_rela = len(rela_gt)
        N_rela_total = N_rela_total + N_rela

        N_obj = len(box_gt)
        N_obj_total = N_obj_total + N_obj

        results[image_id]['N_rlt_gt'] = N_rela
        results[image_id]['N_obj_gt'] = N_obj

        # px1, py1, px2, py2, pname, sx1, sy1, sx2, sy2, sname, ox1, oy1, ox2, oy2, oname, rlt_score
        if image_id not in pred_roidb:
            continue

        curr_pred_roidb = np.array(pred_roidb[image_id])
        if len(curr_pred_roidb) == 0:
            continue

        top_inds = np.argsort(curr_pred_roidb[:, 15])[::-1]
        curr_pred_roidb = curr_pred_roidb[top_inds[:N_recall], :]
        N_pred = len(curr_pred_roidb)

        # [Attention] make hier
        # for d in range(len(curr_pred_roidb)):
        #     if d >= 50:
        #         pre_ind = curr_pred_roidb[d, 4].astype(np.int).tolist()
        #         sbj_ind = curr_pred_roidb[d, 9].astype(np.int).tolist()
        #         obj_ind = curr_pred_roidb[d, 14].astype(np.int).tolist()
        #
        #         pre_node = prenet.get_node_by_index(pre_ind)
        #         sbj_node = objnet.get_node_by_index(sbj_ind)
        #         obj_node = objnet.get_node_by_index(obj_ind)
        #
        #         if len(pre_node.hypers()) > 0:
        #             pre_hyper = pre_node.hypers()[0]
        #             curr_pred_roidb[d, 4] = pre_hyper.index()
        #         if len(sbj_node.hypers()) > 0:
        #             sbj_hyper = sbj_node.hypers()[0]
        #             curr_pred_roidb[d, 9] = sbj_hyper.index()
        #         if len(obj_node.hypers()) > 0:
        #             obj_hyper = obj_node.hypers()[0]
        #             curr_pred_roidb[d, 14] = obj_hyper.index()
        # =====================

        results[image_id]['N_rlt'] = N_pred

        rela_pred = curr_pred_roidb[:, 4].astype(np.int).tolist()
        sub_box_dete = curr_pred_roidb[:, 5:10]
        obj_box_dete = curr_pred_roidb[:, 10:15]
        sub_dete = curr_pred_roidb[:, 9].astype(np.int).tolist()
        obj_dete = curr_pred_roidb[:, 14].astype(np.int).tolist()
        box_det = np.concatenate((sub_box_dete, obj_box_dete))
        box_det = np.unique(box_det, axis=0)

        results[image_id]['N_obj'] = len(box_gt)

        # cal object proposal/detection recall
        img_obj_box_good = 0
        img_obj_det_good = 0

        img_obj_box_rights = [0 for _ in range(len(box_det))]
        img_obj_det_rights = [0 for _ in range(len(box_det))]
        img_obj_box_gt_rights = [0 for _ in range(len(box_gt))]
        img_obj_det_gt_rights = [0 for _ in range(len(box_gt))]

        for b in range(box_gt.shape[0]):
            gt = box_gt[b]
            box_hit = 0
            det_hit = 0

            for o in range(box_det.shape[0]):
                det = box_det[o]
                if compute_iou_each(gt, det) >= box_thr:
                    img_obj_box_rights[o] = 1
                    img_obj_box_gt_rights[b] = 1
                    box_hit = 1
                    if gt[4] == det[4]:
                        img_obj_det_rights[o] = 1
                        img_obj_det_gt_rights[b] = 1
                        det_hit = 1

            img_obj_box_good += box_hit
            img_obj_det_good += det_hit

        results[image_id]['N_obj_box_right'] = sum(img_obj_box_rights)
        results[image_id]['N_obj_det_right'] = sum(img_obj_det_rights)

        results[image_id]['N_obj_box_gt_right'] = sum(img_obj_box_gt_rights)
        results[image_id]['N_obj_det_gt_right'] = sum(img_obj_det_gt_rights)

        N_obj_box_good += img_obj_box_good
        N_obj_det_good += img_obj_det_good

        rela_scores = np.zeros([N_rela, ])
        pre_scores = np.zeros([N_rela, ])

        img_rlt_rights = [0 for _ in range(N_pred)]
        img_rlt_box_rights = [0 for _ in range(N_pred)]
        img_rlt_pair_rights = [0 for _ in range(N_pred)]

        img_rlt_gt_rights = [0 for _ in range(N_rela)]
        img_rlt_box_gt_rights = [0 for _ in range(N_rela)]
        img_rlt_pair_gt_rights = [0 for _ in range(N_rela)]

        for j in range(N_pred):
            # for each relationship prediction
            for k in range(N_rela):
                # for each relationship GT
                if count_gt[k] == 0:
                    continue

                if mode == 'hier':
                    s_iou = compute_iou_each(sub_box_dete[j], sub_box_gt[k])
                    o_iou = compute_iou_each(obj_box_dete[j], obj_box_gt[k])

                    if (s_iou >= box_thr) and (o_iou >= box_thr):
                        count_gt[k] -= 1

                        img_rlt_box_rights[j] = 1
                        img_rlt_box_gt_rights[k] = 1

                        sub_gt_node = objnet.get_node_by_index(sub_gt[k])
                        obj_gt_node = objnet.get_node_by_index(obj_gt[k])
                        pre_gt_node = prenet.get_node_by_index(rela_gt[k])

                        sub_det_node = objnet.get_node_by_index(sub_dete[j])
                        obj_det_node = objnet.get_node_by_index(obj_dete[j])
                        pre_det_node = prenet.get_node_by_index(rela_pred[j])

                        sub_score = sub_gt_node.score(sub_dete[j])
                        obj_score = obj_gt_node.score(obj_dete[j])
                        pre_score = pre_gt_node.score(rela_pred[j])

                        # rela_score = min([sub_score, obj_score, pre_score])
                        if sub_score > 0 and obj_score > 0 and pre_score > 0:
                            rela_score = (sub_score + obj_score + pre_score) / 3.0
                        else:
                            rela_score = 0


                        if sub_score > 0 and obj_score > 0:
                            img_rlt_pair_rights[j] = 1
                            img_rlt_pair_gt_rights[k] = 1

                            if pre_score > 0:
                                img_rlt_rights[j] = 1
                                img_rlt_gt_rights[k] = 1

                                rela_scores[k] = max(rela_scores[k], rela_score)
                                pre_scores[k] = max(pre_scores[k], pre_score)

                        if rela_score >= curr_gt_eval[k, -1]:
                            curr_gt_eval[k, 0] = rela_pred[j]
                            curr_gt_eval[k, 1] = sub_dete[j]
                            curr_gt_eval[k, 2] = obj_dete[j]
                            curr_gt_eval[k, 3] = rela_score
                else:
                    s_iou = compute_iou_each(sub_box_dete[j], sub_box_gt[k])
                    o_iou = compute_iou_each(obj_box_dete[j], obj_box_gt[k])
                    rela_score = 0
                    if (s_iou >= box_thr) and (o_iou >= box_thr):
                        count_gt[k] -= 1
                        img_rlt_box_rights[j] = 1
                        img_rlt_box_gt_rights[k] = 1
                        if (sub_gt[k] == sub_dete[j]) and (obj_gt[k] == obj_dete[j]):
                            img_rlt_pair_rights[j] = 1
                            img_rlt_pair_gt_rights[k] = 1
                            if rela_gt[k] == rela_pred[j]:
                                rela_score = 1
                                img_rlt_rights[j] = 1
                                img_rlt_gt_rights[k] = 1
                                rela_scores[k] = 1
                                pre_scores[k] = 1

                    if rela_score >= curr_gt_eval[k, -1]:
                        curr_gt_eval[k, 0] = rela_pred[j]
                        curr_gt_eval[k, 1] = sub_dete[j]
                        curr_gt_eval[k, 2] = obj_dete[j]
                        curr_gt_eval[k, 3] = rela_score

        results[image_id]['N_rlt_right'] = sum(img_rlt_rights)
        results[image_id]['N_rlt_box_right'] = sum(img_rlt_box_rights)
        results[image_id]['N_rlt_pair_right'] = sum(img_rlt_pair_rights)

        results[image_id]['N_rlt_gt_right'] = sum(img_rlt_gt_rights)
        results[image_id]['N_rlt_box_gt_right'] = sum(img_rlt_box_gt_rights)
        results[image_id]['N_rlt_pair_gt_right'] = sum(img_rlt_pair_gt_rights)

        N_rela_right += np.sum(rela_scores)
        N_pre_right += np.sum(pre_scores)

    det_acc = N_rela_right / N_rela_total
    rec_acc = N_pre_right / N_rela_total
    return det_acc, rec_acc, results, gt_eval
