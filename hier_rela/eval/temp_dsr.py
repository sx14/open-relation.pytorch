import pickle
from ass_fun import *
from global_config import VG_ROOT, VRD_ROOT

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

def show(pred_roidb):
    res = {}
    for img_index in range(len(pred_roidb.keys())):
        pr_curr = pred_roidb[pred_roidb.keys()[img_index]]
        pr_curr = np.array(pr_curr)
        pr_labels = []
        obj_labels = []
        sbj_labels = []
        pr_curr = pr_curr[pr_curr[:,15].argsort()[::-1]]
        if len(pr_curr) > 0:
            if pr_curr[0][15] < 0.5:
                pr_curr = pr_curr[:2]
            else:
                pr_curr = pr_curr[pr_curr[:,15] >= 0.5]
        for i in range(pr_curr.shape[0]):
            pr_cls = pr_curr[i, 4]
            obj_cls = pr_curr[i, 9]
            sbj_cls = pr_curr[i, 14]
            pr_labels.append(prenet.get_node_by_index(int(pr_cls)).name().split('.')[0])
            obj_labels.append(objnet.get_node_by_index(int(obj_cls)).name().split('.')[0])
            sbj_labels.append(objnet.get_node_by_index(int(sbj_cls)).name().split('.')[0])
        res[img_index] = {'dets': pr_curr.tolist(), 'obj_labels': obj_labels, 'sbj_labels': sbj_labels, 'pr_labels': pr_labels}
    return res
