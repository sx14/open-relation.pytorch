import pickle
from ass_fun import *
from global_config import VG_ROOT, VRD_ROOT
from hier_det.show_box import show_boxes
from matplotlib import pyplot as plt

# vrd - vg
dataset = 'vg'
# rela - pre
target = 'rela'
# lu - dsr - vts - ours - dr
method = 'dsr'


if dataset == 'vrd':
    ds_root = VRD_ROOT
    from lib.datasets.vrd.label_hier.obj_hier import objnet
    from lib.datasets.vrd.label_hier.pre_hier import prenet
else:
    ds_root = VG_ROOT
    from lib.datasets.vg200.label_hier.obj_hier import objnet
    from lib.datasets.vg200.label_hier.pre_hier import prenet

if target == 'pre':
    box_thr = 1.0
else:
    box_thr = 0.5

gt_roidb_path = '../gt_rela_roidb_%s.bin' % dataset
gt_roidb = pickle.load(open(gt_roidb_path))

pred_roidb_path = '../%s_box_label_%s_%s.bin' % (target, dataset, method)
pred_roidb = pickle.load(open(pred_roidb_path))

img_root = os.path.join(ds_root, 'JPEGImages')
for img_id in gt_roidb:
    print(img_id)
    gt_curr = gt_roidb[img_id]
    gt_curr = np.array(gt_curr).astype(np.int)
    img_path = os.path.join(img_root, img_id + '.jpg')
    im = plt.imread(img_path)
    if im is None or img_id not in pred_roidb:
        continue

    pr_curr = pred_roidb[img_id]

    gt_boxes = gt_curr[:, 5:9]
    dets_temp = np.copy(gt_boxes)
    dets_temp[:, 2] = gt_boxes[:, 2] - gt_boxes[:, 0]  # width
    dets_temp[:, 3] = gt_boxes[:, 3] - gt_boxes[:, 1]  # height
    gt_labels = []
    for i in range(gt_curr.shape[0]):
        gt_cls = gt_curr[i, 9]
        gt_node = objnet.get_node_by_index(int(gt_cls))
        gt_labels.append(gt_node.name())

    for i in range(gt_curr.shape[0]):
        pre_cls = gt_curr[i, 4]
        sbj_cls = gt_curr[i, 9]
        obj_cls = gt_curr[i, 14]

        pre_node = prenet.get_node_by_index(int(pre_cls))
        sbj_node = objnet.get_node_by_index(int(sbj_cls))
        obj_node = objnet.get_node_by_index(int(obj_cls))

        print('<%s, %s, %s>\t\t[%d, %d, %d, %d]' % (sbj_node.name(), pre_node.name(), obj_node.name(),
                                                    gt_curr[i, 5], gt_curr[i, 6], gt_curr[i, 7], gt_curr[i, 8]))

    show_boxes(im, dets_temp, gt_labels, gt_curr[:, -1], 'all')