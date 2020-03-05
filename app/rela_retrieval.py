import pickle
import time

import numpy as np
import pprint

from global_config import VG_ROOT, VRD_ROOT

# vrd - vg
dataset = 'vg'
top_k = 100

search_concept = 'Frisbee'
print("Let's see %s's relationships" % search_concept)

if dataset == 'vrd':
    ds_root = VRD_ROOT
    from lib.datasets.vrd.label_hier.obj_hier import objnet
else:
    ds_root = VG_ROOT
    from lib.datasets.vg200.label_hier.obj_hier import objnet
    from lib.datasets.vg200.label_hier.pre_hier import prenet

pred_roidb_path = '../hier_rela/rela_box_label_%s_hier_mini.bin' % (dataset)
pred_roidb = pickle.load(open(pred_roidb_path))
print('pred_roidb loaded')

'''
if dataset == 'vg':
    pred_roidb_keys = pred_roidb.keys()[:4000]
    pred_roidb = {key: pred_roidb[key] for key in pred_roidb_keys}
    with open('../hier_rela/rela_box_label_%s_hier_mini.bin' % (dataset), 'wb') as f:
        pickle.dump(pred_roidb, f)
'''


def show_rela(pr_curr):
    print('=====')
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
    pred_roidb[img_id] = pr_curr[:top_k]


def triplet_name(rela):
    sbj = objnet.get_node_by_index(int(rela[9])).name_prefix()
    obj = objnet.get_node_by_index(int(rela[14])).name_prefix()
    pre = prenet.get_node_by_index(int(rela[4])).name_prefix()
    return '%s %s %s' % (sbj, pre, obj)


res = {}
start_tic = time.time()
for img_id in pred_roidb:
    pr_curr = pred_roidb[img_id]
    for rela in pr_curr:
        rela_str = triplet_name(rela)
        if search_concept in rela_str.split(' '):
            if res.has_key(rela_str):
                res[rela_str] += 1
            else:
                res[rela_str] = 1
res = sorted(res.items(), key=lambda x: x[1], reverse=True)
end_tic = time.time()
print('-------')
pprint.pprint(res[:10])
print('time: %s' % (end_tic - start_tic))
