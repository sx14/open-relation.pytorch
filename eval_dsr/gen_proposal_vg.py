import pickle
import cPickle
import scipy.io
import numpy as np
from lib.datasets.vg200.label_hier.obj_hier import objnet


dataset = 'vg'

# load hier det results (0 is background)
raw_labels = objnet.get_raw_labels()
with open('%s/det_roidb_%s.bin' % (dataset, dataset), 'rb') as f:
    dets = pickle.load(f)

# load vg test annotation
gt_path = 'vg/test.pkl'
with open(gt_path, 'rb') as f:
    gt = cPickle.load(f)

img_ids = []
for i in range(len(gt)):
    img_anno = gt[i]
    img_path = img_anno['img_path']
    img_id = img_path.split('/')[-1].split('.')[0]
    img_ids.append(img_id)


# dsr det convert to map
proposals = {}
dsr_boxes = []
dsr_confs = []
dsr_labels = []
dsr_roidb = {}
for img_id in img_ids:
    fast_det = dets[img_id]
    dsr_roidb[img_id] = {}

    pred_boxes = fast_det[:, :4]
    pred_confs = fast_det[:, 5:6]
    pred_labels = fast_det[:, 4:5]

    # to raw label ind
    for p, h_ind in enumerate(pred_labels):
        h_node = objnet.get_node_by_index(int(h_ind[0]))
        find = False
        for ri, raw_label in enumerate(raw_labels):
            if raw_label == h_node.name():
                find = True
                break
        assert find
        pred_labels[p, 0] = ri

    dsr_roidb[img_id]['boxes'] = pred_boxes
    dsr_roidb[img_id]['confs'] = pred_confs
    dsr_roidb[img_id]['cls'] = pred_labels

for img_id in img_ids:

    dsr_boxes.append(dsr_roidb[img_id]['boxes'])
    dsr_confs.append(dsr_roidb[img_id]['confs'])
    dsr_labels.append(dsr_roidb[img_id]['cls'].astype(np.int))

proposals['boxes'] = dsr_boxes
proposals['confs'] = dsr_confs
proposals['cls'] = dsr_labels

dsr_det_path = '%s/proposal.pkl' % dataset
with open(dsr_det_path, 'wb') as f:
    cPickle.dump(proposals, f)
print('gen proposal.pkl done.')

