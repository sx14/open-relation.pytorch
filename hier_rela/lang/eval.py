import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from global_config import HierLabelConfig
from lang_config import train_params, data_config
from lang_dataset import LangDataset
from lib.model.hier_rela.lang.hier_lang import HierLang
from lib.model.hier_rela.lang.hier_lang import order_rank_test as rank_test

dataset = 'vrd'
if dataset == 'vrd':
    from lib.datasets.vrd.label_hier.pre_hier import prenet
elif dataset == 'vg':
    from lib.datasets.vg200.label_hier.pre_hier import prenet

# config and params
obj_config = HierLabelConfig(dataset, 'object')
pre_config = HierLabelConfig(dataset, 'predicate')
pre_label_vec_path = pre_config.label_vec_path()
obj_label_vec_path = obj_config.label_vec_path()

# load test dataset
rlt_path = data_config['test']['raw_rlt_path'] + dataset
test_set = LangDataset(rlt_path, obj_label_vec_path, pre_label_vec_path, prenet)
test_dl = DataLoader(test_set, batch_size=1, shuffle=True)

# model
embedding_dim = test_set.obj_vec_length()

model_save_root = 'output/%s/' % dataset
model = HierLang(embedding_dim * 2, pre_label_vec_path)
weight_path = model_save_root + train_params['best_model_path'] + dataset + '.pth'
if os.path.isfile(weight_path):
    model.load_state_dict(torch.load(weight_path))
    print('Loading weights success.')
model.cuda()
model.eval()

gt_vecs = test_set.get_gt_vecs().float().cuda()
all_raw_inds = set(prenet.get_raw_indexes())
pos_raw_inds = set(prenet.get_raw_indexes()[1:])

N_count = 0
flat_count = 0.0
hier_score_sum = 0.0
raw_score_sum = 0.0

for batch in test_dl:
    N_count += 1

    sbj, pre, obj, pos_neg_inds = batch
    v_sbj = Variable(sbj).float().cuda()
    v_pre = Variable(pre).float().cuda()
    v_obj= Variable(obj).float().cuda()
    with torch.no_grad():
        pre_scores = model(v_sbj, v_obj)

    batch_ranks = rank_test(pre_scores)

    gt_node = prenet.get_node_by_index(pos_neg_inds[0][0])
    gt_label = gt_node.name()
    gt_hyper_inds = gt_node.trans_hyper_inds()

    for ranks in batch_ranks:
        for pre_ind in ranks:
            if pre_ind in pos_raw_inds:
                pre_node = prenet.get_node_by_index(pre_ind)
                if pre_ind == gt_node.index():
                    raw_score_sum += 1

                scr = gt_node.score(pre_ind)
                if scr > 0:
                    flat_count += 1
                    hier_score_sum += scr
                    print('T: %s >>> %s' % (gt_label, pre_node.name()))
                else:
                    print('F: %s >>> %s' % (gt_label, pre_node.name()))
                break

print("==== overall test result ==== ")
print("Rec raw  Acc: %.4f" % (raw_score_sum / N_count))
print("Rec heir Acc: %.4f" % (hier_score_sum / N_count))
print("Rec flat Acc: %.4f" % (flat_count / N_count)) # flat means ignore hierarchy