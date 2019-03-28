import pickle
import cPickle
import scipy.io
import numpy as np

from lib.datasets.vrd.label_hier.pre_hier import prenet
from lib.datasets.vrd.label_hier.obj_hier import objnet

with open('gt_rela_roidb_vrd.bin', 'rb') as f:
    anno = pickle.load(f)

with open('test.pkl', 'rb') as f:
    dsr_test = cPickle.load(f)

dsr_annos = []
raw_obj_labels = objnet.get_raw_labels()
raw_pre_labels = prenet.get_raw_labels()
for i in range(1000):
    if dsr_test[i] is None or dsr_test[i]['img_path'] is None:
        continue

    img_path = dsr_test[i]['img_path']
    img_id = img_path.split('/')[-1].split('.')[0]

    if img_id not in anno:
        continue

    img_anno = anno[img_id]
    if len(img_anno) == 0:
        continue

    img_anno_np = np.array(img_anno)

    dsr_anno = dsr_test[i]
    sbj_dets = img_anno_np[:, 5:10]
    obj_dets = img_anno_np[:, 10:15]
    all_obj_dets = np.concatenate([sbj_dets, obj_dets], axis=0)
    uni_objs, _, inds = np.unique(all_obj_dets, return_index=True, return_inverse=True, axis=0)
    ix1 = inds[:sbj_dets.shape[0]]
    ix2 = inds[sbj_dets.shape[0]:]
    boxes = uni_objs[:, :4]
    h_obj_classes = uni_objs[:, -1]
    h_pre_classes = img_anno_np[:, 4]

    r_obj_classes = []
    for h in range(h_obj_classes.shape[0]):
        h_obj_cls = h_obj_classes[h]
        h_node = objnet.get_node_by_index(int(h_obj_cls))

        find = False
        for ri, raw_obj_label in enumerate(raw_obj_labels):
            if raw_obj_label == h_node.name():
                r_obj_classes.append(ri-1)
                find = True
                break
        assert find
    r_obj_classes = np.array(r_obj_classes).astype(np.int)

    r_pre_classes = []
    for h in range(h_pre_classes.shape[0]):
        h_pre_cls = h_pre_classes[h]
        h_node = prenet.get_node_by_index(int(h_pre_cls))

        find = False
        for ri, raw_pre_label in enumerate(raw_pre_labels):
            if raw_pre_label == h_node.name():
                r_pre_classes.append([ri-1])
                find = True
                break
        assert find

    dsr_anno['boxes'] = boxes
    dsr_anno['classes'] = r_obj_classes
    dsr_anno['ix1'] = ix1
    dsr_anno['ix2'] = ix2
    dsr_anno['rel_classes'] = r_pre_classes
    dsr_test[i] = dsr_anno

with open('test_gt.pkl', 'wb') as f:
    cPickle.dump(dsr_test, f)

