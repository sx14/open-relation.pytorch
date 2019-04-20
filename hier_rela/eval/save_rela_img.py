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


    print('----' + img_id + '----')
    for i in range(img_roidb.shape[0]):
        if img_roidb[i, -1] > 0:
            pre_cls = img_roidb[i, 4]
            sbj_cls = img_roidb[i, 9]
            obj_cls = img_roidb[i, 14]

            pre_label = prenet.get_node_by_index(int(pre_cls)).name().split('.')[0]
            sbj_label = objnet.get_node_by_index(int(sbj_cls)).name().split('.')[0]
            obj_label = objnet.get_node_by_index(int(obj_cls)).name().split('.')[0]

            pre_gt = img_roidb[i, -4]
            sbj_gt = img_roidb[i, -3]
            obj_gt = img_roidb[i, -2]

            pre_gt_label = prenet.get_node_by_index(int(pre_gt))
            sbj_gt_label = objnet.get_node_by_index(int(sbj_gt))
            obj_gt_label = objnet.get_node_by_index(int(obj_gt))

            print('<%s, %s, %s>\t<%s, %s, %s>\t%.2f' % (sbj_gt_label, pre_gt_label, obj_gt_label,
                                                  sbj_label, pre_label, obj_label, img_roidb[i, -1]))

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

if target == 'pre':
    box_thr = 1.0
else:
    box_thr = 0.5

gt_roidb_path = '../gt_rela_roidb_%s.bin' % dataset
gt_roidb = pickle.load(open(gt_roidb_path))

pred_roidb_path = '../%s_box_label_%s_%s.bin' % (target, dataset, method)
pred_roidb = pickle.load(open(pred_roidb_path))

results_path = 'eval_results_%s_%s.bin' % (dataset, method)
results = pickle.load(open(results_path))

img_root = os.path.join(ds_root, 'JPEGImages')

img_id = ''

img_id = '8748523295_c4dd62d359_b' # ride horse
img_id = '6976146955_4a039fc17e_b' # truck !!!
img_id = '6374812041_0af1a67647_b' # tower !!!
img_id = '8376323713_82170250cb_b' # man tower !
img_id = '5684583759_38b776f49f_b' # kitchen .
img_id = '10050248663_2cdb49c115_b'# bus car !
img_id = '3732568219_585f6efb46_b' # train car .
img_id = '9563755622_a8a5e16fb8_b' # table dish .

img_id = '4370372957_91f0753201_b' # man truck !!!!!!!
img_id = '1006083276_0c1a4345fb_o' # people !!!
img_id = '7993454099_f05c523d8a_b' # people
img_id = '4137217003_bc1bd860c0_o' # skateboard
img_id = '7987560060_c4e5b82ae2_o' # drink man figure2
img_id = '7857447002_8d24cba097_b' # vehicle
img_id = '9317704644_c015200b30_b' # cook, figure1
img_id = '7577204808_3c75f97ced_b' # woman back good
img_id = '3131916266_569965d104_b' # man horse figure2
img_id = '8708536141_0dc6882a29_b' # motorcycle india

curr_gt = gt_roidb[img_id]

curr_pr = pred_roidb[img_id]
curr_rs = results[img_id]
img_path = os.path.join(img_root, img_id+'.jpg')
im = plt.imread(img_path)

show_img_relas(curr_pr, curr_rs, im, objnet, prenet, 0.1)


