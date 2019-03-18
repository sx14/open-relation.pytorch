import os
import pickle
import scipy.io
from lib.datasets.vrd.label_hier.obj_hier import objnet
from global_config import PROJECT_ROOT


"""
We use VRD object detection result for now.
"""


det_mat = scipy.io.loadmat('objectDetRCNN.mat')
det_boxes = det_mat['detection_bboxes'][0]
det_labels = det_mat['detection_labels'][0]
det_confs = det_mat['detection_confs'][0]

img_paths_mat = scipy.io.loadmat('imagePath.mat')
img_paths = img_paths_mat['imagePath'][0]

det_roidb = {}
raw_labels = objnet.get_raw_labels()
for i in range(1000):
    img_path = img_paths[i][0]
    img_det_boxes = det_boxes[i]
    img_det_labels = det_labels[i]
    img_det_confs = det_confs[i]

    img_dets = []
    for j in range(img_det_boxes.shape[0]):
        box = img_det_boxes[j]
        conf = img_det_confs[j, 0]
        det = box.tolist()
        det.append(conf)

        label = img_det_labels[j, 0]
        raw_label = raw_labels[label]
        raw_node = objnet.get_node_by_name(raw_label)
        label_ind = raw_node.index()
        det.append(label_ind)
        img_dets.append(det)

    img_id = img_path.split('.')[0]
    det_roidb[img_id] = img_dets

save_path = os.path.join(PROJECT_ROOT, 'hier_rela', 'det_roidb_vrd.bin')
with open(save_path, 'wb') as f:
    pickle.dump(det_roidb, f)
