import os
import pickle
from global_config import PROJECT_ROOT
from hier_det.test_utils import det_recall, load_vrd_det_boxes

dataset = 'vrd'

if dataset == 'vrd':
    from lib.datasets.vrd.label_hier.obj_hier import objnet
else:
    from lib.datasets.vg.label_hier.obj_hier import objnet

if os.path.exists('det_roidb_%s.bin' % dataset):
    with open('det_roidb_%s.bin' % dataset) as f:
        det_roidb = pickle.load(f)
else:
    print('det_roidb_%s.bin not exists.')
    exit(-1)

if os.path.exists('gt_roidb_%s.bin' % dataset):
    with open('gt_roidb_%s.bin' % dataset) as f:
        gt_roidb = pickle.load(f)
else:
    print('gt_roidb_%s.bin not exists' % dataset)
    exit(-1)

# use VRD object detection result
# det_path = os.path.join(PROJECT_ROOT, 'hier_det', 'objectDetRCNN.mat')
# img_path = os.path.join(PROJECT_ROOT, 'hier_det', 'imagePath.mat')
# label_path = os.path.join(PROJECT_ROOT, 'hier_det', 'objectListN.mat')
# det_roidb = load_vrd_det_boxes(det_path, img_path, label_path, objnet)

img_hits = det_recall(gt_roidb, det_roidb, 1000, objnet)


