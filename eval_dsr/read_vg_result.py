import pickle
from lib.datasets.vg200.label_hier.obj_hier import objnet
from lib.datasets.vg200.label_hier.pre_hier import prenet

obj_raw_inds = objnet.get_raw_indexes()
obj_raw_labels = objnet.get_raw_labels()
pre_raw_inds = prenet.get_raw_indexes()[1:]
pre_raw_labels = prenet.get_raw_labels()[1:]

with open('new_test_result.data', 'rb') as f:
    preds = pickle.load(f)

for img_id in preds:
    img_preds = preds[img_id]
    for rlt in img_preds:
        pre_raw_cls = rlt[4]
        sbj_raw_cls = rlt[9]
        obj_raw_cls = rlt[14]

        pre_raw_label = pre_raw_labels[int(pre_raw_cls)]
        sbj_raw_label = obj_raw_labels[int(sbj_raw_cls)]
        obj_raw_label = obj_raw_labels[int(obj_raw_cls)]

        pre_h_cls = pre_raw_inds[int(pre_raw_cls)]
        sbj_h_cls = obj_raw_inds[int(sbj_raw_cls)]
        obj_h_cls = obj_raw_inds[int(obj_raw_cls)]

        sbj_h_node = objnet.get_node_by_index(sbj_h_cls)
        obj_h_node = objnet.get_node_by_index(obj_h_cls)

        rlt[4] = pre_h_cls
        rlt[9] = sbj_h_cls
        rlt[14] = obj_h_cls

with open('rela_box_label_vg_dsr.bin', 'wb') as f:
    pickle.dump(preds, f)
