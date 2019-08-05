import pickle
import numpy as np
from global_config import VRD_ROOT, VG_ROOT


def show_triplet(method, triplet, objnet, prenet):
    pre = prenet.get_node_by_index(int(triplet[0])).name()
    sbj = objnet.get_node_by_index(int(triplet[1])).name()
    obj = objnet.get_node_by_index(int(triplet[2])).name()
    scr = triplet[3]
    print('<%s, %s, %s> (%.2f) %s' % (sbj, pre, obj, scr, method))


dataset = 'vrd'

if dataset == 'vrd':
    ds_root = VRD_ROOT
    from lib.datasets.vrd.label_hier.obj_hier import objnet
    from lib.datasets.vrd.label_hier.pre_hier import prenet
else:
    ds_root = VG_ROOT
    from lib.datasets.vg200.label_hier.obj_hier import objnet
    from lib.datasets.vg200.label_hier.pre_hier import prenet

eval_res_path = 'eval_results_%s_%s.bin'
ours_eval_res_path = eval_res_path % (dataset, 'ours')
with open(ours_eval_res_path) as f:
    ours_eval_res = pickle.load(f)


comparisons = ['vts', 'dsr', 'dr', 'lu']
com_eval_res_list = []
for com in comparisons:
    com_eval_res_path = eval_res_path % (dataset, com)
    with open(com_eval_res_path) as f:
        com_eval_res_list.append(pickle.load(f))

for img_id in ours_eval_res:
    ours = ours_eval_res[img_id]

    recall_ours = np.sum(ours[:, -1]) * 1.0 / ours.shape[0]
    recall_com_max = -1

    for i in range(len(com_eval_res_list)):
        com_eval_res = com_eval_res_list[i][img_id]
        recall = np.sum(com_eval_res[:, -1]) * 1.0 / com_eval_res[:, -1].shape[0]
        recall_com_max = max(recall_com_max, recall)

    if recall_ours > 0.3 and recall_ours > recall_com_max:

        print('==== %s (%.2f) ====' % (img_id, recall_ours))
        for j in range(ours.shape[0]):

            com_max_scr = 0
            for i in range(len(com_eval_res_list)):
                com_max_scr = max(com_max_scr, com_eval_res_list[i][img_id][j, -1])

            if com_max_scr < ours[j][-1] < 1:
                print('----------')
                show_triplet('ours', ours[j], objnet, prenet)
                for i in range(len(com_eval_res_list)):
                    show_triplet(comparisons[i], com_eval_res_list[i][img_id][j], objnet, prenet)
