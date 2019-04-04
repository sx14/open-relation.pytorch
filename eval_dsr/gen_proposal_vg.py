import pickle
import cPickle
import scipy.io
import numpy as np

dataset = 'vg'

with open('%s/gt_rela_roidb_%s.bin' % (dataset, dataset), 'rb') as f:
    anno = pickle.load(f)

with open('%s/det_roidb_%s.bin' % (dataset, dataset), 'rb') as f:
    dets = pickle.load(f)

proposals = {}


# dsr det convert to map
dsr_roidb = {}
dsr_boxes = []
dsr_confs = []
dsr_labels = []
for img_id in anno:

    img_anno = anno[img_id]
    if len(img_anno) == 0:
        continue

    fast_det = dets[img_id]
    dsr_roidb[img_id] = {}

    pred_boxes = fast_det[:, :4]
    pred_confs = fast_det[:, 5:6]
    pred_labels = fast_det[:, 4:5] - 1

    dsr_roidb[img_id]['boxes'] = pred_boxes
    dsr_roidb[img_id]['confs'] = pred_confs
    dsr_roidb[img_id]['cls'] = pred_labels

for img_id in anno:

    img_anno = anno[img_id]
    if len(img_anno) == 0:
        continue

    dsr_boxes.append(dsr_roidb[img_id]['boxes'])
    dsr_confs.append(dsr_roidb[img_id]['confs'])
    dsr_labels.append(dsr_roidb[img_id]['cls'].astype(np.int))

proposals['boxes'] = dsr_boxes
proposals['confs'] = dsr_confs
proposals['cls'] = dsr_labels

dsr_det_path = '%s/proposal_fast.pkl' % dataset
with open(dsr_det_path, 'wb') as f:
    cPickle.dump(proposals, f)
print('gen proposal_fast.pkl done.')

