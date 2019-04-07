import pickle
import cv2
from ass_fun import *
from hier_det.show_box import show_boxes
from global_config import VRD_ROOT, VG_ROOT
# vrd - vg
dataset = 'vrd'
# rela - pre
target = 'rela'
# lu - dsr - vts - ours - dr
method = 'ours'


def show_img_relas(img_roidb, img_results, img, objnet, prenet, thr):
    N_rlt_gt = img_results['N_rlt_gt']
    N_rlt_right = img_results['N_rlt_gt_right']

    if N_rlt_right * 1.0 / N_rlt_gt < thr:
        return


    keep = img_roidb[:, -1] > 0
    hit_roidb = img_roidb[keep, :]
    sbj_dets = hit_roidb[:, 5:10]
    obj_dets = hit_roidb[:, 10:15]

    dets = np.concatenate((sbj_dets, obj_dets), axis=0)
    uni_dets = np.unique(dets, axis=0)
    uni_det_boxes = uni_dets[:, :4]
    uni_det_confs = np.zeros(uni_dets.shape[0])
    uni_det_labels = []
    for i in range(uni_dets.shape[0]):
        uni_det_cls = uni_dets[i, 4]
        label = objnet.get_node_by_index(int(uni_det_cls))
        uni_det_labels.append(label)

    show_boxes(img, uni_det_boxes, uni_det_labels, uni_det_confs)

    for i in range(img_roidb.shape[0]):
        if img_roidb[i, -1] > 0:
            pre_cls = img_roidb[i, 4]
            sbj_cls = img_roidb[i, 9]
            obj_cls = img_roidb[i, 14]

            pre_label = prenet.get_node_by_index(int(pre_cls))
            sbj_label = objnet.get_node_by_index(int(sbj_cls))
            obj_label = objnet.get_node_by_index(int(obj_cls))

            print('<%s, %s, %s>' % (sbj_label, pre_label, obj_label))


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

results_path = '../eval_results_%s_%s.bin' % (dataset, method)
results = pickle.load(open(results_path))

img_root = os.path.join(ds_root, 'JPEGImages')

for img_id in gt_roidb:
    curr_gt = gt_roidb[img_id]
    curr_pr = pred_roidb[img_id]
    curr_rs = results[img_id]
    img_path = os.path.join(img_root, img_id+'.jpg')
    im = cv2.imread(img_path)
    if im is None or curr_gt is None or curr_pr is None or len(curr_gt) == 0 or curr_pr.shape[0] == 0:
        continue

    show_img_relas(curr_pr, curr_rs, im, objnet, prenet, 0.3)


