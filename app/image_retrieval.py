import multiprocessing
import os
import pickle
import time

import cv2
import numpy as np

from global_config import VG_ROOT, VRD_ROOT

print('CPU core count: %d' % multiprocessing.cpu_count())

# vrd - vg
dataset = 'vg'
top_k = 100

if dataset == 'vrd':
    ds_root = VRD_ROOT
    from lib.datasets.vrd.label_hier.obj_hier import objnet
    from lib.datasets.vrd.label_hier.pre_hier import prenet
else:
    ds_root = VG_ROOT
    from lib.datasets.vg200.label_hier.obj_hier import objnet
    from lib.datasets.vg200.label_hier.pre_hier import prenet

pred_roidb_path = '../hier_rela/rela_box_label_%s_hier_mini.bin' % (dataset)
pred_roidb = pickle.load(open(pred_roidb_path))
print('pred_roidb loaded')

N_obj = objnet.label_sum()
N_pre = prenet.label_sum()
obj_sim_mat = np.zeros((N_obj, N_obj), dtype='float32')
pre_sim_mat = np.zeros((N_pre, N_pre), dtype='float32')

for i in range(2, N_obj):
    for j in range(i, N_obj):
        s = objnet.get_node_by_index(i).similarity(objnet.get_node_by_index(j))
        obj_sim_mat[i, j] = s
        obj_sim_mat[j, i] = s

for i in range(2, N_pre):
    for j in range(i, N_pre):
        s = prenet.get_node_by_index(i).similarity(prenet.get_node_by_index(j))
        pre_sim_mat[i, j] = s
        pre_sim_mat[j, i] = s


def score(a, b):
    obj_weight = obj_sim_mat[int(a[9]), int(b[9])]
    sbj_weight = obj_sim_mat[int(a[14]), int(b[14])]
    pre_weight = pre_sim_mat[int(a[4]), int(b[4])]
    if obj_weight == 0 or sbj_weight == 0 or pre_weight == 0:
        return 0
    else:
        return 1.0 / 3 * (obj_weight + sbj_weight + pre_weight) * a[15] * b[15]


if dataset == 'vg':
    pred_roidb_keys = pred_roidb.keys()[:1200]
    pred_roidb = {key: pred_roidb[key] for key in pred_roidb_keys}
    with open('../hier_rela/rela_box_label_%s_hier_mini.bin' % (dataset), 'wb') as f:
        pickle.dump(pred_roidb, f)


def show_rela(pr_curr):
    for i in range(pr_curr.shape[0]):
        pr_cls = pr_curr[i, 4]
        obj_cls = pr_curr[i, 9]
        sbj_cls = pr_curr[i, 14]
        print((objnet.get_node_by_index(int(obj_cls)).name(), prenet.get_node_by_index(int(pr_cls)).name(),
               objnet.get_node_by_index(int(sbj_cls)).name()), pr_curr[i, 15])


# filter and sort
for img_id in pred_roidb:
    pr_curr = pred_roidb[img_id]
    pr_curr = np.array(pr_curr)
    pr_labels = []
    obj_labels = []
    sbj_labels = []
    _, uni_idx = np.unique(pr_curr[:, [4, 9, 14]], axis=0, return_index=True)
    pr_curr = pr_curr[uni_idx]
    for i in range(pr_curr.shape[0]):
        pred_score = pr_curr[i, 15]
        if pred_score >= 30:
            pred_score -= 30
        elif pred_score >= 20:
            pred_score -= 20
        elif pred_score >= 10:
            pred_score -= 10
        pr_curr[i, 15] = pred_score

    pr_curr = pr_curr[pr_curr[:, 15].argsort()[::-1]]

    '''
    if len(pr_curr) > 0:
        if pr_curr[0][15] < 0.4:
            pr_curr = pr_curr[:2]
        else:
            pr_curr = pr_curr[pr_curr[:, 15] >= 0.4]
    '''
    pred_roidb[img_id] = pr_curr[:top_k]

gt_img_index = 90
gt_img_id = pred_roidb.keys()[gt_img_index]
print(gt_img_id)
img_path = os.path.join(ds_root, 'JPEGImages', '%s.jpg' % gt_img_id)
img = cv2.imread(img_path, 1)
cv2.imshow('gt_image', img)
cv2.waitKey(0)
gt = pred_roidb[gt_img_id]
show_rela(gt)
start_tic = time.time()


def func(part_pred_roidb):
    scores = {}
    for img_id in part_pred_roidb:
        pr_curr = part_pred_roidb[img_id]
        sum = 0
        for b in pr_curr:
            s = 0
            for a in gt:
                s += score(a, b)
            sum += s
        scores[img_id] = sum
    return scores


pool = multiprocessing.Pool(processes=12)
jobs = []
for i in range(12):
    p = pool.apply_async(func, ({key: pred_roidb[key] for key in pred_roidb.keys()[i * 100:(i + 1) * 100]},))
    jobs.append(p)

pool.close()
pool.join()

scores = {key: p.get()[key] for p in jobs for key in p.get()}
score_vals = scores.values()
res = np.argsort(np.array(score_vals))[::-1]
end_tic = time.time()
print('-------')
print('time: %s', end_tic - start_tic)
print(res[:30])


def show_preidct(img_indexes):
    for idx in img_indexes:
        img_id = scores.keys()[idx]
        show_rela(pred_roidb[img_id])
        img_path = os.path.join(ds_root, 'JPEGImages', '%s.jpg' % img_id)
        img = cv2.imread(img_path, 1)
        cv2.imshow('pred_image', img)
        k = cv2.waitKey(0)
        if k == ord('e'):
            cv2.destroyAllWindows()
            break


show_preidct(res[:30])
