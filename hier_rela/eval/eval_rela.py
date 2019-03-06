import pickle
from ass_fun import *


dataset = 'vrd'

target = 'rela'
# target = 'pre'

gt_roidb_path = '../gt_box_label_%s.bin' % dataset
test_roidb = pickle.load(open(gt_roidb_path))

pred_roidb_path = '../%s_box_label_%s.bin' % (target, dataset)
pred_roidb = pickle.load(open(pred_roidb_path))

R50, num_right50 = rela_recall(test_roidb, pred_roidb, 50)
R100, num_right100 = rela_recall(test_roidb, pred_roidb, 100)

print('R50: {0}, R100: {1}'.format(R50, R100))
