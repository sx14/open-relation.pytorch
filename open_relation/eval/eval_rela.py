import pickle
import numpy as np
from ass_fun import *
from open_relation.global_config import project_root
from open_relation.dataset.dataset_config import DatasetConfig

dataset = 'vrd'
dataset_config = DatasetConfig(dataset)

gt_roidb_path = os.path.join(dataset_config.extra_config['predicate'].prepare_root, 'test_box_label.bin')
test_roidb = pickle.load(open(gt_roidb_path))

pre_roidb_path = os.path.join(project_root, 'open_relation', 'output', dataset, 'pre_box_label.bin')
pred_roidb = pickle.load(open(pre_roidb_path))

R50, num_right50 = rela_recall(test_roidb, pred_roidb, 50)
R100, num_right100 = rela_recall(test_roidb, pred_roidb, 100)

print('R50: {0}, R100: {1}'.format(R50, R100))
