import pickle
from ass_fun import *


dataset = 'vrd'

if dataset == 'vrd':
    from lib.datasets.vrd.label_hier.obj_hier import objnet
    from lib.datasets.vrd.label_hier.pre_hier import prenet
else:
    from lib.datasets.vg1000.label_hier.obj_hier import objnet
    from lib.datasets.vg1000.label_hier.pre_hier import prenet


# target = 'rela'
target = 'pre'

gt_roidb_path = '../gt_rela_roidb_%s.bin' % dataset
gt_roidb = pickle.load(open(gt_roidb_path))

pred_roidb_path = '../%s_box_label_%s.bin' % (target, dataset)
pred_roidb = pickle.load(open(pred_roidb_path))

rela_R50, pre_R50,  num_right50 = rela_recall('hier', gt_roidb, pred_roidb, 50, objnet, prenet)
rela_R100, pre_R100, num_right100 = rela_recall('hier', gt_roidb, pred_roidb, 100, objnet, prenet)

print('rela R50: %.4f, rela R100: %.4f' % (rela_R50, rela_R100))
print('pre R50: %.4f, pre R100: %.4f' % (pre_R50, pre_R100))