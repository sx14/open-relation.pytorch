import os
import pickle
import numpy as np
from global_config import PROJECT_ROOT

raw_det_path = os.path.join(PROJECT_ROOT, 'hier_det', 'det_roidb_vg_100.bin')
with open(raw_det_path) as f:
    raw_det_roidb = pickle.load(f)

det_roidb = {}
thresh = 0.2
for img_id in raw_det_roidb:
    img_dets = raw_det_roidb[img_id]
    img_det_scores = img_dets[:, -1]
    inds = np.where(img_det_scores > thresh)[0]
    good_dets = img_dets[inds]
    det_roidb[img_id] = good_dets

det_path = os.path.join(PROJECT_ROOT, 'hier_rela', 'det_roidb_vg_good.bin')
with open(raw_det_path, 'wb') as f:
    pickle.dump(det_roidb, det_path)
