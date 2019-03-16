import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from lang_dataset import LangDataset
from lang_config import train_params, data_config
from lib.model.hier_rela.lang.hier_lang import HierLang
from lib.model.hier_rela.lang.hier_lang import order_rank_test as rank_test
from lib.datasets.vrd.label_hier.pre_hier import prenet
from global_config import HierLabelConfig


dataset = 'vrd'
# hyper params
obj_config = HierLabelConfig(dataset, 'object')
pre_config = HierLabelConfig(dataset, 'predicate')
pre_label_vec_path = pre_config.label_vec_path()
obj_label_vec_path = obj_config.label_vec_path()
rlt_path = data_config['test']['raw_rlt_path']+dataset
test_set = LangDataset(rlt_path, obj_label_vec_path, pre_label_vec_path, prenet)
test_dl = DataLoader(test_set, batch_size=1, shuffle=True)

# model
embedding_dim = test_set.obj_vec_length()

model_save_root = 'output/%s/' % dataset
model = HierLang(embedding_dim * 2, pre_label_vec_path)
weight_path = model_save_root + train_params['best_model_path']+dataset+'.pth'
if os.path.isfile(weight_path):
    model.load_state_dict(torch.load(weight_path))
    print('Loading weights success.')
model.cuda()
model.eval()

batch_num = 0
pos_num = 0
neg_num = 0

gt_vecs = test_set.get_gt_vecs().float().cuda()
all_raw_inds = set(prenet.get_raw_indexes())
pos_raw_inds = set(prenet.get_raw_indexes()[1:])

acc = 0.0
pos_acc = 0.0
neg_acc = 0.0

for batch in test_dl:

    sbj1, pre1, obj1, pos_neg_inds = batch
    v_sbj1 = Variable(sbj1).float().cuda()
    v_pre1 = Variable(pre1).float().cuda()
    v_obj1 = Variable(obj1).float().cuda()
    with torch.no_grad():
        pre_scores1 = model(v_sbj1, v_obj1)

    batch_ranks = rank_test(pre_scores1, gt_vecs)

    gt_node = prenet.get_node_by_index(pos_neg_inds[0][0])
    gt_label = gt_node.name()
    gt_hyper_inds = gt_node.trans_hyper_inds()

    if gt_node.index() == 0:
        neg_num += 1
    else:
        pos_num += 1
    batch_num += 1

    for ranks in batch_ranks:


        # print('\n===== GT: %s =====' % gt_label)
        # for gt_h_ind in gt_hyper_inds:
        #     gt_h_node = prenet.get_node_by_index(gt_h_ind)
            # print(gt_h_node.name())
        # print('===== predict =====')
        t = 0
        if gt_node.index() == 0:
            for pre_ind in ranks:
                if pre_ind in all_raw_inds:
                    pre_node = prenet.get_node_by_index(pre_ind)
                    if pre_ind == gt_node.index():
                        neg_acc += 1
                        t = 1
                        print('T: %s >>> %s' % (gt_label, pre_node.name()))
                    else:
                        print('F: %s >>> %s' % (gt_label, pre_node.name()))
                    break
        else:
            for pre_ind in ranks:
                if pre_ind in pos_raw_inds:
                    pre_node = prenet.get_node_by_index(pre_ind)
                    if pre_ind == gt_node.index():
                        pos_acc += 1
                        t = 1
                        print('T: %s >>> %s' % (gt_label, pre_node.name()))
                    else:
                        print('F: %s >>> %s' % (gt_label, pre_node.name()))
                    break
        acc += t

print('\nraw acc >>> %.4f' % (acc / batch_num))
print('\npos acc >>> %.4f' % (pos_acc / pos_num))
print('\nneg acc >>> %.4f' % (neg_acc / neg_num))

