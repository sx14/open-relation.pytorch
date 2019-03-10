import os
import numpy as np
import pickle

import torch

from lib.model.nms.nms_wrapper import nms
from lib.model.hier_utils.tree_infer import my_infer, raw2cond_prob
from hier_det.test_utils import det_recall, load_vrd_det_boxes
from global_config import PROJECT_ROOT
from hier_det.test_utils import nms_dets, nms_dets1

dataset = 'vrd'

if dataset == 'vrd':
    from lib.datasets.vrd.label_hier.obj_hier import objnet
else:
    from lib.datasets.vg.label_hier.obj_hier import objnet

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

# process
N_img = len(det_roidb.keys())

nms_roidb = {}
for i, img_id in enumerate(det_roidb.keys()):
    print('nms [%d/%d]' % (N_img, i + 1))
    img_dets = det_roidb[img_id]

    img_det_scores = img_dets[:, 4:]
    img_det_cond_p = raw2cond_prob(objnet, img_det_scores)
    img_dets[:, 4:] = img_det_cond_p

    curr_rois = nms_dets1(img_dets, 100, objnet)
    nms_roidb[img_id] = curr_rois


with open('det_infer_roidb_%s.bin' % dataset, 'wb') as f:
    pickle.dump(nms_roidb, f)

img_hits = det_recall(gt_roidb, nms_roidb, 100, objnet)


