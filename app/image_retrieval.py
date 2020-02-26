import os
import pickle
import random
import time

import cv2
import numpy as np
from datasketch import MinHash, MinHashLSH

from global_config import VG_ROOT, VRD_ROOT
from utils.extractor import extract_triplet
from utils.query_expansion import expand_query
from utils.diversity import rerank

# vrd - vg
dataset = 'vg'
top_k = 100

search_rela = extract_triplet('vessel on table')
print(search_rela)

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

mHash_query = MinHash()
expanded_query = expand_query([search_rela])
for q in expanded_query:
    mHash_query.update(q)

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

# construct index
lsh = MinHashLSH(threshold=0)


def triplet_name(rela):
    sbj = objnet.get_node_by_index(int(rela[9])).name_prefix()
    obj = objnet.get_node_by_index(int(rela[14])).name_prefix()
    pre = prenet.get_node_by_index(int(rela[4])).name_prefix()
    return '%s %s %s' % (sbj, pre, obj)


for img_id in pred_roidb:
    mHash = MinHash()
    pr_curr = pred_roidb[img_id]
    for rela in pr_curr:
        mHash.update(triplet_name(rela))
    lsh.insert(img_id, mHash)

expanded_query_set = set(expanded_query)

start_tic = time.time()
res = lsh.query(mHash_query)  # image_ids
hash_tic = time.time()
filtered_roidb = {}
for img_id in res:
    filtered_roidb[img_id] = set([triplet_name(rela) for rela in pred_roidb[img_id]]).intersection(expanded_query_set)
filter_tic = time.time()
values = np.array(filtered_roidb.values())
sorted_idxs = np.argsort(np.array([len(relas) for relas in values]))[::-1]  # first rerank by semantic similarity
res = rerank(values[sorted_idxs])

end_tic = time.time()
print('-------')
print('found %d images, hash time: %s, filter time: %s, rerank time: %s' % (
    len(res), hash_tic - start_tic, filter_tic - hash_tic, end_tic - filter_tic))


def show_predict(img_idxs):
    for img_idx in img_idxs:
        img_id = filtered_roidb.keys()[sorted_idxs[img_idx]]
        print(filtered_roidb[img_id])
        img_path = os.path.join(ds_root, 'JPEGImages', '%s.jpg' % img_id)
        img = cv2.imread(img_path, 1)
        cv2.imshow('pred_image', img)
        k = cv2.waitKey(0)
        if k == ord('e'):
            cv2.destroyAllWindows()
            break


show_predict(res[:30])


def show_next_random_image():
    random_img_id = random.choice(pred_roidb.keys())
    show_rela(pred_roidb[random_img_id])
    img_path = os.path.join(ds_root, 'JPEGImages', '%s.jpg' % random_img_id)
    img = cv2.imread(img_path, 1)
    cv2.imshow('pred_image', img)
    k = cv2.waitKey(0)
    if k == ord('e'):
        cv2.destroyAllWindows()

# show_next_random_image()
