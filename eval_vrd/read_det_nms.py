import os
import pickle
import cPickle

import cv2
import numpy as np
import scipy.io

import torch
from torch import Tensor

from lib.model.nms.nms_wrapper import nms
from lib.datasets.vrd.label_hier.obj_hier import objnet
from lib.datasets.tools.show_box import show_boxes
from global_config import PROJECT_ROOT, VRD_ROOT


"""
We use VRD object detection result for now.
"""


def confirm(img_path, dets):
    im = cv2.imread(img_path)
    cls = []
    boxes = []
    for i in range(len(dets)):
        label = dets[i][-2]
        n = objnet.get_node_by_index(int(label))
        cls.append(n.name())

        box = dets[i][:4]
        w = box[2] - box[0]
        h = box[3] - box[1]
        box[2] = w
        box[3] = h
        boxes.append(box)
    show_boxes(im, boxes, cls)


# load dsr det
with open('proposal.pkl', 'rb') as fid:
    proposals = cPickle.load(fid)
    det_boxes = proposals['boxes']
    det_labels = proposals['cls']
    det_confs = proposals['confs']

# load image paths
img_paths_mat = scipy.io.loadmat('imagePath.mat')
img_paths = img_paths_mat['imagePath'][0]

N = 0
N_remove = 0
det_roidb = {}
raw_labels = objnet.get_raw_labels()[1:]
for i in range(1000):
    img_path = img_paths[i][0]
    img_det_boxes = det_boxes[i]
    img_det_labels = det_labels[i]
    img_det_confs = det_confs[i]

    if img_det_boxes.shape[0] > 0:
        # do NMS
        img_det_nms = np.concatenate((img_det_boxes, img_det_confs, img_det_labels), axis=1)
        img_det_nms = torch.from_numpy(img_det_nms).cuda()
        keep = nms(img_det_nms, 0.8)
        keep_num = keep.shape[0]
        N_remove += img_det_boxes.shape[0] - keep_num
        img_det_nms = img_det_nms[keep.view(-1).long()].cpu().numpy()

        img_det_boxes = img_det_nms[:, :4]
        img_det_confs = img_det_nms[:, 4:5]
        img_det_labels = img_det_nms[:, 5:6]

        det_boxes[i] = img_det_boxes
        det_labels[i] = img_det_labels.astype(np.int)
        det_confs[i] = img_det_confs

    img_dets = []
    for j in range(img_det_boxes.shape[0]):
        N += 1
        box = img_det_boxes[j]
        det = box.tolist()

        label = img_det_labels[j, 0]
        raw_label = raw_labels[int(label)]
        raw_node = objnet.get_node_by_name(raw_label)
        label_ind = raw_node.index()
        det.append(label_ind)

        conf = img_det_confs[j, 0]
        det.append(conf)

        img_dets.append(det)

    img_id = img_path.split('.')[0]
    det_roidb[img_id] = np.array(img_dets)

# save ours det
save_path = os.path.join(PROJECT_ROOT, 'hier_rela', 'det_roidb_nms_vrd.bin')
with open(save_path, 'wb') as f:
    pickle.dump(det_roidb, f)

# save dsr det
proposals['boxes'] = det_boxes
proposals['cls'] = det_labels
proposals['confs'] = det_confs
with open('proposal_nms.pkl', 'wb') as fid:
    cPickle.dump(proposals, fid)


print('det num: %d' % N)
print('rev num: %d' % N_remove)

# img_id = det_roidb.keys()[1]
# img_path = os.path.join(VRD_ROOT, 'JPEGImages', img_id+'.jpg')
# dets = det_roidb[img_id]
# confirm(img_path, dets)
