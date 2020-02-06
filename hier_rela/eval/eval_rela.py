import pickle

from eval_utils import rela_recall

# vrd - vg
dataset = 'vrd'
# rela - pre
target = 'pre'

save_result = False

if dataset == 'vrd':
    from lib.datasets.vrd.label_hier.obj_hier import objnet
    from lib.datasets.vrd.label_hier.pre_hier import prenet
else:
    from lib.datasets.vg200.label_hier.obj_hier import objnet
    from lib.datasets.vg200.label_hier.pre_hier import prenet

if target == 'pre':
    box_thr = 1.0  # pre use gt boxes, so iou == 1
else:
    box_thr = 0.5

gt_roidb_path = '../gt_rela_roidb_%s.bin' % dataset
gt_roidb = pickle.load(open(gt_roidb_path))

pred_roidb_path = '../%s_box_label_%s_hier.bin' % (target, dataset)
pred_roidb = pickle.load(open(pred_roidb_path))

# analysis
recall_Ns = [50, 100]
for recall_N in recall_Ns:
    rela_acc, pre_acc, results, gt_eval = rela_recall('hier', gt_roidb, pred_roidb, recall_N, objnet, prenet, alpha=2,
                                                      box_thr=box_thr)
    N_rlt_gt_all = 0.0
    N_rlt_pred_all = 0.0

    N_rlt_box_right_all = 0.0
    N_rlt_pair_right_all = 0.0
    N_rlt_right_all = 0.0

    N_rlt_box_gt_right_all = 0.0
    N_rlt_pair_gt_right_all = 0.0
    N_rlt_gt_right_all = 0.0

    for img in results:
        N_rlt_gt_all += results[img]['N_rlt_gt']

        N_rlt_pred_all += results[img]['N_rlt']

        N_rlt_box_right_all += results[img]['N_rlt_box_right']
        N_rlt_pair_right_all += results[img]['N_rlt_pair_right']
        N_rlt_right_all += results[img]['N_rlt_right']

        N_rlt_box_gt_right_all += results[img]['N_rlt_box_gt_right']
        N_rlt_pair_gt_right_all += results[img]['N_rlt_pair_gt_right']
        N_rlt_gt_right_all += results[img]['N_rlt_gt_right']

    print('==== relationship(%d) ====' % recall_N)
    print('proposal recall: \t%.5f' % (N_rlt_box_gt_right_all / N_rlt_gt_all))
    print('proposal precision: \t%.5f' % (N_rlt_box_right_all / N_rlt_pred_all))
    print('detection recall: \t%.5f' % (N_rlt_pair_gt_right_all / N_rlt_gt_all))
    print('detection precision: \t%.5f' % (N_rlt_pair_right_all / N_rlt_pred_all))
    print('relationship B-Recall: \t%.5f' % (N_rlt_gt_right_all / N_rlt_gt_all))
    print('relationship precision: \t%.5f\n' % (N_rlt_right_all / N_rlt_pred_all))

    print('rela HR%s: %.5f' % (recall_N, rela_acc))
    print('pre HR%s: %.5f' % (recall_N, pre_acc))

    if save_result:
        pickle.dump(gt_eval, open('eval_results_%s_recall%s.bin' % (dataset, recall_N), 'wb'))
