import scipy.io
import numpy as np


gt = scipy.io.loadmat('gt.mat')
gt_tuples = gt['gt_tuple_label'][0]
gt_obj_boxes = gt['gt_obj_bboxes'][0]
gt_sbj_boxes = gt['gt_sub_bboxes'][0]

gt_counts = []
for i in range(1000):
    im_tuples = gt_tuples[i]
    im_sbj_clses = im_tuples[:, 0:1]
    im_obj_clses = im_tuples[:, 2:3]
    im_sbj_boxes = np.concatenate((gt_sbj_boxes[i], im_sbj_clses), axis=1)
    im_obj_boxes = np.concatenate((gt_obj_boxes[i], im_obj_clses), axis=1)
    im_pairs = np.concatenate((im_sbj_boxes, im_obj_boxes), axis=1)

    im_counts = [0 for _ in range(im_pairs.shape[0])]

    for j in range(im_pairs.shape[0]):
        ins_pair1 = im_pairs[j]
        for k in range(im_pairs.shape[0]):
            ins_pair2 = im_pairs[k]
            if np.sum(ins_pair1 - ins_pair2) == 0:
                im_counts[j] += 1

    gt_counts.append(im_counts)

gt_counts = np.array([gt_counts])
gt['gt_counts'] = gt_counts
scipy.io.savemat('gt1.mat', gt)
