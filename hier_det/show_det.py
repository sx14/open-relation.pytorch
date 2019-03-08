import os
import pickle
import cv2
import numpy as np
from hier_det.show_box import show_boxes
from global_config import VRD_ROOT


dataset = 'vrd'
det_roidb_path = 'det_infer_roidb_vrd.bin'
det_roidb = pickle.load(open(det_roidb_path))

img_root = os.path.join(VRD_ROOT, 'JPEGImages')
for img_id in det_roidb:
    img_path = os.path.join(img_root, img_id+'.jpg')
    im = cv2.imread(img_path)
    dets = det_roidb[img_id]
    dets_temp = np.copy(dets)
    dets_temp[:, 2] = dets[:, 2] - dets[:, 0]   # width
    dets_temp[:, 3] = dets[:, 3] - dets[:, 1]   # height
    labels = dets[:, 4]
    confs = dets[:, 5]
    show_boxes(im, dets_temp[:, :4], confs)