import os
import pickle
import cv2
import numpy as np
from hier_det.show_box import show_boxes
from global_config import VRD_ROOT, PROJECT_ROOT, VG_ROOT


dataset = 'vg'
det_roidb_path = os.path.join(PROJECT_ROOT, 'eval_dsr', 'vg', 'det_roidb_%s_flat.bin' % dataset)
det_roidb = pickle.load(open(det_roidb_path))

if dataset == 'vg':
    DS_ROOT = VG_ROOT
    from lib.datasets.vg200.label_hier.obj_hier import objnet

else:
    DS_ROOT = VRD_ROOT
    from lib.datasets.vrd.label_hier.obj_hier import objnet

img_root = os.path.join(DS_ROOT, 'JPEGImages')
raw_labels = objnet.get_raw_labels()
for img_id in det_roidb:
    img_path = os.path.join(img_root, img_id+'.jpg')
    im = cv2.imread(img_path)
    dets = det_roidb[img_id]
    dets_temp = np.copy(dets)
    dets_temp[:, 2] = dets[:, 2] - dets[:, 0]   # width
    dets_temp[:, 3] = dets[:, 3] - dets[:, 1]   # height
    cls_inds = dets[:, 4]
    cls_labels = []

    for ind in cls_inds:
        cls_labels.append(raw_labels[ind])

    confs = dets[:, 5]
    show_boxes(im, dets_temp[:, :4], cls_labels, confs)

