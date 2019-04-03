import pickle
from ass_fun import *


dataset = 'vrd'

if dataset == 'vrd':
    from lib.datasets.vrd.label_hier.obj_hier import objnet
    from lib.datasets.vrd.label_hier.pre_hier import prenet
else:
    from lib.datasets.vg200.label_hier.obj_hier import objnet
    from lib.datasets.vg200.label_hier.pre_hier import prenet


# target = 'rela'
target = 'pre'

gt_roidb_path = '../gt_rela_roidb_%s.bin' % dataset
gt_roidb = pickle.load(open(gt_roidb_path))

pred_roidb_path = '../%s_box_label_%s_dsr.bin' % (target, dataset)
pred_roidb = pickle.load(open(pred_roidb_path))

rela_R50, pre_R50, results_R50 = rela_recall('hier', gt_roidb, pred_roidb, 50, objnet, prenet, alpha=2, box_thr=0.5)
rela_R100, pre_R100, results_R100 = rela_recall('hier', gt_roidb, pred_roidb, 100, objnet, prenet, alpha=2, box_thr=0.5)


# analysis
recall_Ns = [50, 100]
all_results = [results_R50, results_R100]
for i, results in enumerate(all_results):
    N_obj_gt_all = 0.0
    N_rlt_gt_all = 0.0

    N_obj_pred_all = 0.0
    N_rlt_pred_all = 0.0

    N_obj_box_right_all = 0.0
    N_obj_det_right_all = 0.0

    N_obj_box_gt_right_all = 0.0
    N_obj_det_gt_right_all = 0.0

    N_rlt_box_right_all = 0.0
    N_rlt_pair_right_all = 0.0
    N_rlt_right_all = 0.0

    N_rlt_box_gt_right_all = 0.0
    N_rlt_pair_gt_right_all = 0.0
    N_rlt_gt_right_all = 0.0

    for img in results:
        N_obj_gt_all += results[img]['N_obj_gt']
        N_rlt_gt_all += results[img]['N_rlt_gt']

        N_obj_pred_all += results[img]['N_obj']
        N_rlt_pred_all += results[img]['N_rlt']

        N_obj_box_right_all += results[img]['N_obj_box_right']
        N_obj_det_right_all += results[img]['N_obj_det_right']

        N_obj_box_gt_right_all += results[img]['N_obj_box_gt_right']
        N_obj_det_gt_right_all += results[img]['N_obj_det_gt_right']

        N_rlt_box_right_all += results[img]['N_rlt_box_right']
        N_rlt_pair_right_all += results[img]['N_rlt_pair_right']
        N_rlt_right_all += results[img]['N_rlt_right']

        N_rlt_box_gt_right_all += results[img]['N_rlt_box_gt_right']
        N_rlt_pair_gt_right_all += results[img]['N_rlt_pair_gt_right']
        N_rlt_gt_right_all += results[img]['N_rlt_gt_right']

    print('==== object(%d) ====' % recall_Ns[i])
    print('proposal recall: \t%.4f' % (N_obj_box_gt_right_all / N_obj_gt_all))
    print('proposal precision: \t%.4f' % (N_obj_box_right_all / N_obj_pred_all))
    print('detection recall: \t%.4f' % (N_obj_det_gt_right_all / N_obj_gt_all))
    print('detection precision: \t%.4f\n' % (N_obj_det_right_all / N_obj_pred_all))

    print('==== relationship(%d) ====' % recall_Ns[i])
    print('proposal recall: \t%.4f' % (N_rlt_box_gt_right_all / N_rlt_gt_all))
    print('proposal precision: \t%.4f' % (N_rlt_box_right_all / N_rlt_pred_all))
    print('detection recall: \t%.4f' % (N_rlt_pair_gt_right_all / N_rlt_gt_all))
    print('detection precision: \t%.4f' % (N_rlt_pair_right_all / N_rlt_pred_all))
    print('relationship recall: \t%.4f' % (N_rlt_gt_right_all / N_rlt_gt_all))
    print('relationship precision: \t%.4f\n' % (N_rlt_right_all / N_rlt_pred_all))



print('rela R50: %.4f, rela R100: %.4f' % (rela_R50, rela_R100))
print('pre R50: %.4f, pre R100: %.4f' % (pre_R50, pre_R100))
