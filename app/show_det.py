import numpy as np
from show_box_det import show_boxes
from global_config import VRD_ROOT, PROJECT_ROOT, VG_ROOT



dataset = 'vrd'

if dataset == 'vg':
    DS_ROOT = VG_ROOT
    from lib.datasets.vg200.label_hier.obj_hier import objnet

else:
    DS_ROOT = VRD_ROOT
    from lib.datasets.vrd.label_hier.obj_hier import objnet

def show(det_roidb):
    res = {}
    for img_index in det_roidb:
        dets = det_roidb[img_index]
        dets = dets[dets[:,5].argsort()[::-1]]
        res[img_index] = {"dets": dets.tolist()}
        cls_inds = dets[:, 4]
        cls_labels = []
        for ind in cls_inds:
            cls_labels.append(objnet.get_all_labels()[int(ind)].split('.')[0])
        res[img_index]["labels"] = cls_labels
    return res