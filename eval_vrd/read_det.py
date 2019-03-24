import os
import pickle

import cv2
import numpy as np
import scipy.io
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
        label = dets[i][4]
        n = objnet.get_node_by_index(int(label.tolist()))
        cls.append(n.name())

        box = dets[i][:4]
        w = box[2] - box[0]
        h = box[3] - box[1]
        box[2] = w
        box[3] = h
        boxes.append(box)


    show_boxes(im, boxes, cls)


det_mat = scipy.io.loadmat('objectDetRCNN.mat')
det_boxes = det_mat['detection_bboxes'][0]
det_labels = det_mat['detection_labels'][0]
det_confs = det_mat['detection_confs'][0]

img_paths_mat = scipy.io.loadmat('imagePath.mat')
img_paths = img_paths_mat['imagePath'][0]

det_roidb = {}
raw_labels = objnet.get_raw_labels()
N_det = 0.0
for i in range(1000):
    img_path = img_paths[i][0]
    img_det_boxes = det_boxes[i]
    img_det_labels = det_labels[i]
    img_det_confs = det_confs[i]

    # x1, y1, x2 ,y2, cls, conf
    img_dets = []
    for j in range(img_det_boxes.shape[0]):
        N_det += 1
        box = img_det_boxes[j]
        det = box.tolist()

        label = img_det_labels[j, 0]
        raw_label = raw_labels[label]
        raw_node = objnet.get_node_by_name(raw_label)
        label_ind = raw_node.index()
        det.append(label_ind)

        conf = img_det_confs[j, 0]
        det.append(conf)

        img_dets.append(det)

    img_id = img_path.split('.')[0]
    det_roidb[img_id] = np.array(img_dets)

print('Avg det per img: %.4f' % (N_det / len(det_roidb.keys())))

save_path = os.path.join(PROJECT_ROOT, 'hier_rela', 'det_roidb_vrd.bin')
with open(save_path, 'wb') as f:
    pickle.dump(det_roidb, f)

# img_id = det_roidb.keys()[5]
# img_path = os.path.join(VRD_ROOT, 'JPEGImages', img_id+'.jpg')
# dets = det_roidb[img_id]
# confirm(img_path, dets)
