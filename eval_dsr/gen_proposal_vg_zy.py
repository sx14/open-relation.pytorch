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


for img_id in dets:
    hier_det = dets[img_id]

    pred_boxes = hier_det[:, :4]
    pred_confs = hier_det[:, 5:6]
    pred_labels = hier_det[:, 4:5]

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



dsr_det_path = '%s/det_roidb_%s_raw.bin' % (dataset, dataset)
with open(dsr_det_path, 'wb') as f:
    cPickle.dump(dets, f)
print('gen det_roidb_vg_raw.bin done.')

