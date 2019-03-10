# coding=utf-8

import os
import numpy as np
import cv2
import torch
from lib.model.nms.nms_wrapper import nms


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


def det_recall(gt_roidb, pred_roidb, N_recall, objnet):
    N_obj_total = 0.0

    N_obj_box_good = 0.0
    N_obj_det_good = 0.0
    S_obj_det_score = 0.0

    num_right = {}
    for image_id in gt_roidb:
        num_right[image_id] = 0
        # x1, y1, x2, y2, name
        curr_gt_roidb = np.array(gt_roidb[image_id])

        if len(curr_gt_roidb) == 0:
            continue
        box_gt = curr_gt_roidb[:, :4]

        N_obj = len(box_gt)
        N_obj_total = N_obj_total + N_obj

        # x1, y1, x2, y2, name, score
        if image_id not in pred_roidb:
            print(image_id)
            continue

        curr_pred_roidb = np.array(pred_roidb[image_id])
        if len(curr_pred_roidb) == 0:
            continue

        # sort predictions
        scr_dete = curr_pred_roidb[:, 5]
        ranked_inds = np.argsort(scr_dete)[::-1]
        curr_pred_roidb = curr_pred_roidb[ranked_inds]

        if curr_pred_roidb.shape[0] > N_recall:
            curr_pred_roidb = curr_pred_roidb[:N_recall]

        gt_max_scores = [0 for _ in range(box_gt.shape[0])]
        for b in range(box_gt.shape[0]):
            gt = curr_gt_roidb[b]
            box_hit = 0
            det_hit = 0
            det_scr = 0
            for o in range(curr_pred_roidb.shape[0]):
                if det_hit == 1:
                    break
                det = curr_pred_roidb[o]
                if compute_iou_each(gt, det[:4]) > 0.5:
                    box_hit = 1
                    if gt[4] == det[4]:
                        det_hit = 1
                    det_scr = max([det_scr, objnet.get_node_by_index(int(gt[4].tolist())).score(int(det[4].tolist()))])

            gt_max_scores[b] = det_scr
            N_obj_box_good += box_hit
            N_obj_det_good += det_hit
            S_obj_det_score += det_scr
        num_right[image_id] = sum(gt_max_scores) / box_gt.shape[0]

    print('Proposal recall@%d: %.4f' % (N_recall, N_obj_box_good / N_obj_total))
    print('Detection flat recall@%d: %.4f' % (N_recall, N_obj_det_good / N_obj_total))
    print('Detection hier recall@%d: %.4f' % (N_recall, S_obj_det_score / N_obj_total))
    return num_right


def load_vrd_det_boxes(vrd_box_path, vrd_img_path, vrd_label_path, objnet):
    import scipy.io as sio
    import copy

    vrd_boxes = sio.loadmat(vrd_box_path)['detection_bboxes'][0]
    vrd_confs = sio.loadmat(vrd_box_path)['detection_confs'][0]
    vrd_labels = sio.loadmat(vrd_box_path)['detection_labels'][0]
    vrd_ind2label = sio.loadmat(vrd_label_path)['objectListN'][0]
    vrd_imgs = sio.loadmat(vrd_img_path)['imagePath'][0]
    det_roidb = dict()

    for i in range(vrd_imgs.shape[0]):
        img = vrd_imgs[i][0]
        img_id = img.split('.')[0]
        boxes = vrd_boxes[i]
        confs = vrd_confs[i]
        labels = copy.deepcopy(vrd_labels[i])

        for e in range(labels.shape[0]):
            # break
            l = labels[e, 0]
            name = vrd_ind2label[l-1][0]
            node = objnet.get_node_by_name(name)
            labels[e, 0] = node.index()

        roidb = np.concatenate((boxes, labels, confs), axis=1)
        det_roidb[img_id] = roidb
    return det_roidb


def nms_dets(img_dets, max_det_num, objnet):
    from lib.datasets.vrd.label_hier.obj_hier import objnet

    pred_boxes = torch.from_numpy(img_dets[:, :4]).cuda()
    scores = torch.from_numpy(img_dets[:, 4:]).cuda()
    leaf_cls = objnet.get_raw_indexes()

    N_classes = scores.shape[0]
    all_boxes = [[] for _ in xrange(len(leaf_cls))]
    all_scrs = [[] for _ in xrange(len(leaf_cls))]
    all_dets = [[] for _ in xrange(len(leaf_cls))]

    threshold = -2
    empty_box = np.transpose(np.array([[], [], [], []]), (1, 0))
    empty_det = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    empty_cls = np.transpose(np.array([[] for _ in range(N_classes)]), (1, 0))


    for j in xrange(1, len(leaf_cls)):
        leaf_c = leaf_cls[j]
        # for each leaf class
        inds = torch.nonzero(scores[:, leaf_c] > threshold).view(-1)
        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:, leaf_c][inds]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[inds, :]
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, 0.3)
            cls_dets = cls_dets[keep.view(-1).long()]
            all_dets[j] = cls_dets.cpu().numpy()

            box_scores = scores[inds, :]
            box_scores = box_scores[order]
            box_scores = box_scores[keep.view(-1).long()]
            cls_boxes = cls_boxes[order]
            cls_boxes = cls_boxes[keep.view(-1).long()]
            all_boxes[j] = cls_boxes.cpu().numpy()
            all_scrs[j] = box_scores.cpu().numpy()

        else:
            all_boxes[j] = empty_box
            all_scrs[j] = empty_cls
            all_dets[j] = empty_det


    # Limit to max_det_num detections *over all classes*
    nms_det_scores = []
    image_scores = np.hstack([all_dets[j][:, -1] for j in xrange(1, len(leaf_cls))])
    if len(image_scores) > max_det_num:
        image_thresh = np.sort(image_scores)[-max_det_num]
        for j in xrange(1, len(leaf_cls)):
            keep = np.where(all_dets[j][:, -1] >= image_thresh)[0]
            all_dets[j] = all_dets[j][keep, :]
            all_boxes[j] = all_boxes[j][keep, :]
            all_scrs[j] = all_scrs[j][keep, :]
            if all_dets[j].shape[0] > 0:
                box_scores = np.concatenate((all_boxes[j], all_scrs[j]), axis=1)
                nms_det_scores += box_scores.tolist()
    else:
        for j in xrange(1, len(leaf_cls)):
            all_dets[j] = all_dets[j][:, :]
            all_boxes[j] = all_boxes[j][:, :]
            all_scrs[j] = all_scrs[j][:, :]
            if all_dets[j].shape[0] > 0:
                box_scores = np.concatenate((all_boxes[j], all_scrs[j]), axis=1)
                nms_det_scores += box_scores.tolist()
    return np.array(nms_det_scores)


def nms_dets1(img_dets, max_det_num, objnet):
    from lib.datasets.vrd.label_hier.obj_hier import objnet

    # 300 proposal
    # 300 * N_classes scores
    pred_boxes = torch.from_numpy(img_dets[:, :4]).cuda()
    scores = torch.from_numpy(img_dets[:, 4:]).cuda()

    N_classes = scores.shape[1]
    all_dets = [[] for _ in xrange(N_classes)]
    empty_det = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    for j in xrange(1, N_classes):
        # for each class
        cls_node = objnet.get_node_by_index(j)
        if len(cls_node.hypers()) == 0:
            thresh = 0.9
        else:
            thresh = 1.0 / len(cls_node.hypers()[0].children()) + 0.1
        inds = torch.nonzero(scores[:, j] > thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[inds, :]
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, 0.3)
            cls_dets = cls_dets[keep.view(-1).long()]
            cls_dets = cls_dets.cpu().numpy()

            # score = conditional probability * info ratio
            cls_info_ratio = objnet.get_node_by_index(j).info_ratio(objnet.label_sum())
            cls_info_ratio = np.array(cls_info_ratio)
            cls_info_ratio = np.sqrt(cls_info_ratio)
            cls_dets[:, 4] = cls_dets[:, 4] * cls_info_ratio
            all_dets[j] = cls_dets

        else:
            all_dets[j] = empty_det

    # Limit to max_det_num detections *over all classes*
    all_dets_list = []
    image_scores = np.hstack([all_dets[j][:, -1] for j in xrange(1, N_classes)])
    if len(image_scores) > max_det_num:
        image_thresh = np.sort(image_scores)[-max_det_num]
        for j in xrange(1, N_classes):
            keep = np.where(all_dets[j][:, -1] >= image_thresh)[0]
            all_dets[j] = all_dets[j][keep, :]

            if keep.shape[0] > 0:
                cls_dets = all_dets[j]
                cls_inds = np.zeros((cls_dets.shape[0], 1))
                cls_inds[:, :] = j

                cls_rois = np.concatenate((cls_dets[:, :4], cls_inds, cls_dets[:, 4][:, np.newaxis]), 1)
                all_dets_list += cls_rois.tolist()

    return np.array(all_dets_list)



















