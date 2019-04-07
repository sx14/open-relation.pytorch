import pickle
from matplotlib import pyplot as plt
from ass_fun import *
from hier_det.show_box import show_boxes
from global_config import VRD_ROOT, VG_ROOT
# vrd - vg
dataset = 'vrd'
# rela - pre
target = 'rela'
# lu - dsr - vts - ours - dr
methods = ['lu', 'dsr', 'vts', 'dr', 'ours']
results = []
for method in methods:
    results_path = 'eval_results_%s_%s.bin' % (dataset, method)
    result = pickle.load(open(results_path))
    results.append(result)

if dataset == 'vrd':
    ds_root = VRD_ROOT
    from lib.datasets.vrd.label_hier.obj_hier import objnet
    from lib.datasets.vrd.label_hier.pre_hier import prenet
else:
    ds_root = VG_ROOT
    from lib.datasets.vg200.label_hier.obj_hier import objnet
    from lib.datasets.vg200.label_hier.pre_hier import prenet


for img_id in results[0]:
    img_results = []

    for i in range(len(methods)):
        img_results.append(results[i][img_id])

    max_recall = 0
    max_recall_id = -1
    for i in range(len(img_results)):
        N_rlt_gt = img_results[i]['N_rlt_gt']
        N_rlt_right = img_results[i]['N_rlt_gt_right']
        recall = N_rlt_right * 1.0 / N_rlt_gt
        if recall > max_recall:
            max_recall = recall
            max_recall_id = i

    if max_recall_id == len(methods) - 1:
        print('%s: %.2f' % (img_id, max_recall))





