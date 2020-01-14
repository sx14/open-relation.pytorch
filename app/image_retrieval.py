import pickle
from global_config import VG_ROOT, VRD_ROOT
import numpy as np
import os
import cv2

# vrd - vg
dataset = 'vrd'
# rela - pre
target = 'rela'
# lu - dsr - vts - ours - dr
method = 'ours'


if dataset == 'vrd':
    ds_root = VRD_ROOT
    from lib.datasets.vrd.label_hier.obj_hier import objnet
    from lib.datasets.vrd.label_hier.pre_hier import prenet
else:
    ds_root = VG_ROOT
    from lib.datasets.vg200.label_hier.obj_hier import objnet
    from lib.datasets.vg200.label_hier.pre_hier import prenet

pred_roidb_path = '../hier_rela/%s_box_label_%s_%s_hier_01.bin' % (target, dataset, method)
pred_roidb = pickle.load(open(pred_roidb_path))

def show_rela(pr_curr):
    for i in range(pr_curr.shape[0]):
        pr_cls = pr_curr[i, 4]
        obj_cls = pr_curr[i, 9]
        sbj_cls = pr_curr[i, 14]
        print((objnet.get_node_by_index(int(obj_cls)).name(),prenet.get_node_by_index(int(pr_cls)).name(),objnet.get_node_by_index(int(sbj_cls)).name()), pr_curr[i,15])

# filter and sort
for img_id in pred_roidb:
    pr_curr = pred_roidb[img_id]
    pr_curr = np.array(pr_curr)
    pr_labels = []
    obj_labels = []
    sbj_labels = []
    for i in range(pr_curr.shape[0]):
        pred_score = pr_curr[i, 15]
        if pred_score >= 30:
            pred_score -= 30
        elif pred_score >= 20:
            pred_score -= 20
        elif pred_score >= 10:
            pred_score -= 10
        pr_curr[i, 15] = pred_score

    pr_curr = pr_curr[pr_curr[:,15].argsort()[::-1]]

    if len(pr_curr) > 0:
        if pr_curr[0][15] < 0.4:
            pr_curr = pr_curr[:2]
        else:
            pr_curr = pr_curr[pr_curr[:,15] >= 0.4]
    _,uni_idx  = np.unique(pr_curr[:, [4,9,14]], axis=0, return_index=True)
    pred_roidb[img_id] = pr_curr[uni_idx]

print objnet.get_node_by_index(1).is_partial_order(objnet.get_node_by_index(9))
gt_img_id = pred_roidb.keys()[79]
print(gt_img_id)
img_path = os.path.join(VRD_ROOT, 'JPEGImages', '%s.jpg' % gt_img_id)
img = cv2.imread(img_path, 1)
cv2.imshow('gt_image', img)
cv2.waitKey(0)
gt = pred_roidb[gt_img_id]
show_rela(gt)

def score(a,b):
    obj_a = objnet.get_node_by_index(int(a[9]))
    sbj_a = objnet.get_node_by_index(int(a[14]))
    pre_a = prenet.get_node_by_index(int(a[4]))

    obj_b = objnet.get_node_by_index(int(b[9]))
    sbj_b = objnet.get_node_by_index(int(b[14]))
    pre_b = prenet.get_node_by_index(int(b[4]))

    obj_weight = max(obj_a.is_partial_order(obj_b),obj_b.is_partial_order(obj_a))
    sbj_weight = max(sbj_a.is_partial_order(sbj_b),sbj_b.is_partial_order(sbj_a))
    pre_weight = max(pre_a.is_partial_order(pre_b),pre_b.is_partial_order(pre_a))
    return obj_weight * sbj_weight * pre_weight * a[15] * b[15]
scores = []
for img_id in pred_roidb:
    pr_curr = pred_roidb[img_id]
    sum = 0
    for b in pr_curr:
        s = 0
        for a in gt:
            s += score(a,b)
        sum += s
    scores.append(sum)
res = np.argsort(np.array(scores))[::-1]
print('-------')
print(np.sort(np.array(scores))[::-1][:30])
predict_img_id = pred_roidb.keys()[res[0]]
# show_rela(pred_roidb[predict_img_id])

def show_preidct(img_indexes):
    for idx in img_indexes:
        img_id = pred_roidb.keys()[idx]
        img_path = os.path.join(VRD_ROOT, 'JPEGImages', '%s.jpg' % img_id)
        img = cv2.imread(img_path, 1)
        cv2.imshow('pred_image', img)
        k = cv2.waitKey(0)
        if k == ord('e'):
            cv2.destroyAllWindows()
            break
        # print(img_path)

show_preidct(res[:30])