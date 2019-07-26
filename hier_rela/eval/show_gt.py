import pickle
from matplotlib import pyplot as plt
from ass_fun import *
from hier_det.show_box import show_boxes
from global_config import VRD_ROOT, VG_ROOT
from nms import py_cpu_nms

# vrd - vg
dataset = 'vg'



def show_img_relas(gt_roidb, img, objnet, prenet):

    keep = gt_roidb[:, -1] > 0
    hit_roidb = gt_roidb[keep, :]
    sbj_dets = hit_roidb[:, 5:10]
    obj_dets = hit_roidb[:, 10:15]

    dets = np.concatenate((sbj_dets, obj_dets), axis=0)

    uni_dets = np.unique(dets, axis=0)
    keep = py_cpu_nms(uni_dets, 0.7)
    uni_dets = uni_dets[keep]

    uni_det_boxes = uni_dets[:, :4]
    uni_det_confs = np.zeros(uni_dets.shape[0])
    uni_det_labels = []
    for i in range(uni_dets.shape[0]):
        uni_det_cls = uni_dets[i, 4]
        label = objnet.get_node_by_index(int(uni_det_cls)).name()
        # uni_det_labels.append(label.split('.')[0])
        uni_det_labels.append(label)

    good = False
    for i in range(gt_roidb.shape[0]):
        pre_gt = gt_roidb[i, 4]
        sbj_gt = gt_roidb[i, 9]
        obj_gt = gt_roidb[i, 14]

        pre_gt_label = prenet.get_node_by_index(int(pre_gt))
        sbj_gt_label = objnet.get_node_by_index(int(sbj_gt))
        obj_gt_label = objnet.get_node_by_index(int(obj_gt))
        if pre_gt_label.name() == 'standing next to':
            good = True
            print('<%s, %s, %s>' % (sbj_gt_label, pre_gt_label, obj_gt_label))

    if not good:
        return
    dets_temp = np.copy(uni_det_boxes)
    dets_temp[:, 2] = uni_det_boxes[:, 2] - uni_det_boxes[:, 0]  # width
    dets_temp[:, 3] = uni_det_boxes[:, 3] - uni_det_boxes[:, 1]  # height
    show_boxes(img, dets_temp, uni_det_labels, uni_det_confs, 'all')


if dataset == 'vrd':
    ds_root = VRD_ROOT
    from lib.datasets.vrd.label_hier.obj_hier import objnet
    from lib.datasets.vrd.label_hier.pre_hier import prenet

else:
    ds_root = VG_ROOT
    from lib.datasets.vg200.label_hier.obj_hier import objnet
    from lib.datasets.vg200.label_hier.pre_hier import prenet


gt_roidb_path = '../gt_rela_roidb_%s.bin' % dataset
gt_roidb = pickle.load(open(gt_roidb_path))


img_root = os.path.join(ds_root, 'JPEGImages')

for img_id in gt_roidb:

    print('===== %s =====' % img_id)
    curr_gt = gt_roidb[img_id]
    curr_gt = np.array(curr_gt)

    img_path = os.path.join(img_root, img_id+'.jpg')
    im = plt.imread(img_path)

    show_img_relas(curr_gt, im, objnet, prenet)


