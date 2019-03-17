import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from pro_dataset import ProDataset
from lang_config import train_params, data_config
from lib.model.hier_rela.lang.hier_lang import HierLang
from lib.model.hier_rela.lang.hier_lang import order_rank_test as rank_test
from lib.datasets.vrd.label_hier.pre_hier import prenet
from global_config import HierLabelConfig

def ext_box_feat(gt_relas):
    # spacial feats
    sbj_boxes = gt_relas[:, 5:9]
    obj_boxes = gt_relas[:, 10:14]

    sbj_boxes_w = sbj_boxes[:, 2] - sbj_boxes[:, 0]
    sbj_boxes_h = sbj_boxes[:, 3] - sbj_boxes[:, 1]

    obj_boxes_w = obj_boxes[:, 2] - obj_boxes[:, 0]
    obj_boxes_h = obj_boxes[:, 3] - obj_boxes[:, 1]

    sbj_tx = (sbj_boxes[:, 0] - obj_boxes[:, 0]) / sbj_boxes_w
    sbj_ty = (sbj_boxes[:, 1] - obj_boxes[:, 1]) / sbj_boxes_h
    sbj_tw = torch.log(sbj_boxes_w / obj_boxes_w)
    sbj_th = torch.log(sbj_boxes_h / obj_boxes_h)

    obj_tx = (obj_boxes[:, 0] - sbj_boxes[:, 0]) / obj_boxes_w
    obj_ty = (obj_boxes[:, 1] - sbj_boxes[:, 1]) / obj_boxes_h
    obj_tw = torch.log(obj_boxes_w / sbj_boxes_w)
    obj_th = torch.log(obj_boxes_h / sbj_boxes_h)

    sbj_feat_vecs = torch.cat([sbj_tx.unsqueeze(1), sbj_ty.unsqueeze(1),
                               sbj_tw.unsqueeze(1), sbj_th.unsqueeze(1)], dim=1)
    obj_feat_vecs = torch.cat([obj_tx.unsqueeze(1), obj_ty.unsqueeze(1),
                               obj_tw.unsqueeze(1), obj_th.unsqueeze(1)], dim=1)

    return sbj_feat_vecs, obj_feat_vecs


dataset = 'vrd'
# hyper params
obj_config = HierLabelConfig(dataset, 'object')
pre_config = HierLabelConfig(dataset, 'predicate')
pre_label_vec_path = pre_config.label_vec_path()
obj_label_vec_path = obj_config.label_vec_path()
rlt_path = data_config['test']['raw_rlt_path']+dataset
test_set = ProDataset(rlt_path, obj_label_vec_path, pre_label_vec_path, prenet)
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

    sbj1, pre1, obj1, pos_neg_inds, rlt = batch
    v_sbj1 = Variable(sbj1).float().cuda()
    v_obj1 = Variable(obj1).float().cuda()
    v_rlt = Variable(rlt).float().cuda()
    v_sbj_box, v_obj_box = ext_box_feat(v_rlt)
    v_sbj1 = torch.cat([v_sbj1, v_sbj_box], dim=1)
    v_obj1 = torch.cat([v_obj1, v_obj_box], dim=1)
    
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

