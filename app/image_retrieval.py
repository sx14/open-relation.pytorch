import os
import pickle
import random
import time

import cv2

from database.rela_db import RelaDB
from rela_retrieval import search_by_concept
from utils.extractor import extract_triplet
from utils.query_expansion import expand_query

# vrd - vg
dataset = 'vg'
top_k = 100
rela_db = RelaDB('./database/scripts/rela_db.db')

image_root = '/Users/lioder/Downloads/VG_100K_2'
from lib.datasets.vg200.label_hier.obj_hier import objnet
from lib.datasets.vg200.label_hier.pre_hier import prenet


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, size):
        return self.fileobj.read(size).encode()

    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


def triplet_name(rela):
    sbj = objnet.get_node_by_index(int(rela[1])).name_prefix()
    obj = objnet.get_node_by_index(int(rela[2])).name_prefix()
    pre = prenet.get_node_by_index(int(rela[0])).name_prefix()
    return '%s %s %s' % (sbj, pre, obj)


pred_roidb_path = '../hier_rela/rela_box_label_%s_hier_pure.bin' % (dataset)
pred_roidb = pickle.load(StrToBytes(open(pred_roidb_path)), encoding='bytes')
print('image db loaded')

'''
if dataset == 'vg':
    pred_roidb_keys = pred_roidb.keys()[:4000]
    pred_roidb = {key: pred_roidb[key] for key in pred_roidb_keys}
    with open('../hier_rela/rela_box_label_%s_hier_mini.bin' % (dataset), 'wb') as f:
        pickle.dump(pred_roidb, f)

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
    pred_roidb[img_id] = pr_curr[:top_k, [4,9,14,15]]

with open('../hier_rela/rela_box_label_%s_hier_pure.bin' % (dataset), 'wb') as f:
    pickle.dump(pred_roidb, f)


with lsh.insertion_session() as session:
    for img_id in pred_roidb:
        mHash = MinHash()
        pr_curr = pred_roidb[img_id]
        for rela in pr_curr:
            mHash.update(triplet_name(rela))
        session.insert(img_id, mHash)
# lsh = MinHashLSH(threshold=0, storage_config={
#     'type': 'redis',
#     'basename': b'lsh_20200311_6',
#     'redis': {'host': 'localhost', 'port': 6379}
# })

'''


# with lsh.insertion_session() as session:
#     for img_id in pred_roidb:
#         mHash = MinHash()
#         pr_curr = pred_roidb[img_id]
#         for rela in pr_curr:
#             if objnet.get_node_by_index(int(rela[1])) is not None and objnet.get_node_by_index(
#                     int(rela[2])) is not None and prenet.get_node_by_index(int(rela[0])) is not None:
#                 rela_str = triplet_name(rela)
#                 mHash.update(rela_str.encode())
#             else:
#                 print('wrong rela %f %f %F' %(rela[0], rela[1], rela[2]))
#         session.insert(img_id, mHash)


def show_rela(pr_curr):
    print('=====')
    for i in range(pr_curr.shape[0]):
        pr_cls = pr_curr[i, 0]
        obj_cls = pr_curr[i, 2]
        sbj_cls = pr_curr[i, 1]
        print((objnet.get_node_by_index(int(obj_cls)).name(), prenet.get_node_by_index(int(pr_cls)).name(),
               objnet.get_node_by_index(int(sbj_cls)).name()), pr_curr[i, 3])


def search(text):
    if text.strip() == "":
        return {"imageIds": [], "image2rela": {}}
    search_rela = extract_triplet(text)
    print(search_rela)

    if search_rela[0] != '' and search_rela[1] == '' and search_rela[2] == '':
        return search_by_concept(search_rela[0])

    if objnet.get_node_by_name_prefix(search_rela[0]) is None or objnet.get_node_by_name_prefix(
            search_rela[2]) is None or prenet.get_node_by_name_prefix(search_rela[1]) is None:
        return {"imageIds": [], "image2rela": {}}

    sbjs, pres, objs = expand_query([search_rela])

    res = [tuple[0].decode() for tuple in rela_db.find_images_by_rela(sbjs, pres, objs)]  # [(image_id),...]
    image_ids = list(set(res))
    return {
        "imageIds": image_ids[:30],
        "image2rela": {}
    }


'''
def show_predict(img_idxs):
    for img_idx in img_idxs:
        img_id = filtered_roidb.keys()[sorted_idxs[img_idx]]
        print(filtered_roidb[img_id])
        img_path = os.path.join(ds_root, 'JPEGImages', '%s.jpg' % img_id)
    # print('found %d images, hash time: %s, filter time: %s, rerank time: %s' % (
    #     len(res), hash_tic - start_tic))
    # image_ids = [str(list(filtered_roidb.keys())[sorted_idxs[image_idx]], encoding='utf-8') for image_idx in res[:100]]
    image_ids = list(set(res))
    return {
        "imageIds": image_ids[:30],
        "image2rela": {}
    }
'''


def show_search_result(img_ids):
    for img_id in img_ids:
        img_path = os.path.join(image_root, '%s.jpg' % img_id)
        if not os.path.exists(img_path):
            print("sorry, the image should be viewed on server")
            continue
        img = cv2.imread(img_path, 1)
        cv2.imshow('image', img)
        k = cv2.waitKey(0)
        if k == ord('e'):
            cv2.destroyAllWindows()
            break


def show_next_random_image():
    random_img_id = random.choice(pred_roidb.keys())
    show_rela(pred_roidb[random_img_id])
    img_path = os.path.join(image_root, '%s.jpg' % random_img_id)
    img = cv2.imread(img_path, 1)
    cv2.imshow('pred_image', img)
    k = cv2.waitKey(0)
    if k == ord('e'):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print(search('bowl on carnivore')["imageIds"])
    rela_db.close()
