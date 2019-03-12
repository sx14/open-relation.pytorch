import os
import numpy as np
import pickle

import torch

from lib.model.nms.nms_wrapper import nms
from lib.model.hier_utils.tree_infer import my_infer
from hier_det.test_utils import det_recall, load_vrd_det_boxes
from global_config import PROJECT_ROOT
from hier_det.test_utils import nms_dets

dataset = 'vrd'

if dataset == 'vrd':
    from lib.datasets.vrd.label_hier.obj_hier import objnet
else:
    from lib.datasets.vg1000.label_hier.obj_hier import objnet

if os.path.exists('det_roidb_%s.bin' % dataset):
    with open('det_roidb_%s.bin' % dataset) as f:
        det_roidb = pickle.load(f)
    print('Load det_roidb_%s.bin successfully.' % dataset)
else:
    print('det_roidb_%s.bin not exists.' % dataset)
    exit(-1)

if os.path.exists('gt_roidb_%s.bin' % dataset):
    with open('gt_roidb_%s.bin' % dataset) as f:
        gt_roidb = pickle.load(f)
    print('Load gt_roidb_%s.bin successfully.' % dataset)
else:
    print('gt_roidb_%s.bin not exists' % dataset)
    exit(-1)

# use VRD object detection result
# det_path = os.path.join(PROJECT_ROOT, 'hier_det', 'objectDetRCNN.mat')
# img_path = os.path.join(PROJECT_ROOT, 'hier_det', 'imagePath.mat')
# label_path = os.path.join(PROJECT_ROOT, 'hier_det', 'objectListN.mat')
# det_roidb = load_vrd_det_boxes(det_path, img_path, label_path, objnet)

# process
N_img = len(det_roidb.keys())
if os.path.exists('det_infer_roidb_%s.bin' % dataset):
    with open('det_infer_roidb_%s.bin' % dataset, 'rb') as f:
        det_roidb = pickle.load(f)
else:
    for i, img_id in enumerate(det_roidb.keys()):
        print('nms [%d/%d]' % (N_img, i + 1))
        img_dets = det_roidb[img_id]
        my_dets = nms_dets(img_dets, 100, objnet)
        det_roidb[img_id] = my_dets

    for i, img_id in enumerate(det_roidb.keys()):
        print('infer [%d/%d]' % (N_img, i+1))
        my_dets = det_roidb[img_id]
        pred_boxes = my_dets[:, :4]
        pred_scores = my_dets[:, 4:]

        infer_labels = np.zeros((pred_scores.shape[0], 1))
        infer_scores = np.zeros((pred_scores.shape[0], 1))

        for mmm in range(pred_scores.shape[0]):
            top2 = my_infer(objnet, pred_scores[mmm])
            infer_labels[mmm] = top2[0][0]
            infer_scores[mmm] = top2[0][1]

        my_dets = np.concatenate([pred_boxes, infer_labels, infer_scores], 1)
        my_dets = torch.from_numpy(my_dets).cuda()
        keep = nms(my_dets, 0.5)
        my_dets = my_dets[keep.view(-1).long()]
        my_dets = my_dets.cpu().data.numpy()
        det_roidb[img_id] = my_dets

    with open('det_infer_roidb_%s.bin' % dataset, 'wb') as f:
        pickle.dump(det_roidb, f)



img_hits = det_recall(gt_roidb, det_roidb, 100, objnet)


