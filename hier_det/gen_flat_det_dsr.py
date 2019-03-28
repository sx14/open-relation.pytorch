import os
import pickle
import cPickle
import scipy.io
import numpy as np

from global_config import PROJECT_ROOT


# DSR proposal container
dsr_proposal_path = os.path.join(PROJECT_ROOT, 'eval_dsr', 'proposal.pkl')
with open(dsr_proposal_path, 'rb') as fid:
    dsr_dets = cPickle.load(fid)
dsr_boxes = dsr_dets['boxes']
dsr_confs = dsr_dets['confs']
dsr_labels = dsr_dets['cls']

# load image paths
img_path_path = os.path.join(PROJECT_ROOT, 'eval_dsr', 'imagePath.mat')
img_paths_mat = scipy.io.loadmat(img_path_path)
img_paths = img_paths_mat['imagePath'][0]

# load ours flat detection(vrd proposal)
det_roidb_flat_path = os.path.join(PROJECT_ROOT, 'hier_rela', 'det_roidb_flat_vrd.bin')
with open(det_roidb_flat_path, 'rb') as f:
    pred_roidb = pickle.load(f)

# dsr det convert to map
dsr_roidb = {}
for i in range(len(img_paths)):
    img_id = img_paths[i][0].split('.')[0]
    dsr_roidb[img_id] = {}
    dsr_roidb[img_id]['boxes'] = dsr_boxes[i]
    dsr_roidb[img_id]['confs'] = dsr_confs[i]
    dsr_roidb[img_id]['cls'] = dsr_labels[i]

# replace dsr det with faster-rcnn det
for img_id in dsr_roidb:
    if img_id not in pred_roidb:
        continue
    pred_det = pred_roidb[img_id]
    pred_boxes = pred_det[:, :4]
    pred_confs = pred_det[:, 5:6]
    pred_labels = pred_det[:, 4:5] - 1

    dsr_roidb[img_id]['boxes'] = pred_boxes
    dsr_roidb[img_id]['confs'] = pred_confs
    dsr_roidb[img_id]['cls'] = pred_labels

for i in range(len(img_paths)):
    img_id = img_paths[i][0].split('.')[0]
    dsr_boxes[i] = dsr_roidb[img_id]['boxes']
    dsr_confs[i] = dsr_roidb[img_id]['confs']
    dsr_labels[i] = dsr_roidb[img_id]['cls'].astype(np.int)

dsr_dets['boxes'] = dsr_boxes
dsr_dets['confs'] = dsr_confs
dsr_dets['cls'] = dsr_labels

dsr_det_path = os.path.join(PROJECT_ROOT, 'eval_dsr', 'proposal_fast.pkl')
with open(dsr_det_path, 'wb') as f:
    cPickle.dump(dsr_dets, f)
print('gen proposal_fast.pkl done.')
