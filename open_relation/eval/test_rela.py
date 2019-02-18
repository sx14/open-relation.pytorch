import os
import copy
import pickle
import h5py
import numpy as np
import cv2
import torch

from open_relation import global_config
from open_relation.model.predicate.model import PredicateVisual
from open_relation.dataset.dataset_config import DatasetConfig
from open_relation.train.train_config import hyper_params
from open_relation.dataset.vrd.label_hier.pre_hier import prenet
from open_relation.dataset.vrd.label_hier.obj_hier import objnet
from open_relation.language.infer.model import RelationEmbedding
from open_relation.language.infer.lang_config import train_params
from relationship.ext_cnn_feat import ext_cnn_feat

def gen_rela_conds(det_roidb):
    rela_cands = dict()
    for img_id in det_roidb:
        rela_cands_temp = []
        rois = det_roidb[img_id]
        for i, sbj in enumerate(rois):
            for j, obj in enumerate(rois):
                if i == j:
                    continue
                px1 = min(sbj[0], obj[0])
                py1 = min(sbj[1], obj[1])
                px2 = max(sbj[2], obj[2])
                py2 = max(sbj[3], obj[3])
                rela_temp = [px1, py1, px2, py2, -1] + sbj.tolist() + obj.tolist()
                rela_cands_temp.append(rela_temp)
        rela_cands[img_id] = rela_cands_temp
    return rela_cands


def gen_prediction(scores, raw_label_inds, mode='raw'):
    scores = scores.cpu().data
    ranked_inds = np.argsort(scores).tolist()
    ranked_inds.reverse()  # descending
    pred_ind = -1
    pred_score = -1
    for ind in ranked_inds:
        if ind in raw_label_inds:
            pred_ind = ind
            pred_score = scores[ind]
            break
    assert pred_ind > -1
    return pred_ind, pred_score


dataset = 'vrd'

# use = 'vis_only'
# use = 'lan_only'
use = 'vis_lan'

score_mode = 'raw'
# score_mode = 'hier'

# prepare data
dataset_config = DatasetConfig(dataset)
pre_config = hyper_params[dataset]['predicate']
obj_config = hyper_params[dataset]['object']
det_roidb_path = dataset_config.extra_config['object'].det_box_path
det_roidb = pickle.load(open(det_roidb_path))
rela_cands = gen_rela_conds(det_roidb)

pre_label_vec_path = pre_config['label_vec_path']
label_embedding_file = h5py.File(pre_label_vec_path, 'r')
pre_label_vecs = np.array(label_embedding_file['label_vec'])
label_embedding_file.close()

obj_label_vec_path = obj_config['label_vec_path']
label_embedding_file = h5py.File(obj_label_vec_path, 'r')
obj_label_vecs = np.array(label_embedding_file['label_vec'])
label_embedding_file.close()

# prepare label maps
pre_raw2path = prenet.raw2path()
pre_label2index = prenet.label2index()
pre_index2label = prenet.index2label()
raw_pre_inds = set([pre_label2index[l] for l in prenet.get_raw_labels()])

obj_raw2path = objnet.raw2path()
obj_label2index = objnet.label2index()
obj_index2label = objnet.index2label()
raw_obj_inds = set([obj_label2index[l] for l in objnet.get_raw_labels()])




# load visual model with best weights
vmodel_best_weights_path = pre_config['best_weight_path']
vmodel = PredicateVisual(obj_config['visual_d'], obj_config['hidden_d'],
                         obj_config['embedding_d'], obj_config['label_vec_path'],
                         obj_config['best_weight_path'],
                         pre_config['visual_d'], pre_config['hidden_d'],
                         pre_config['embedding_d'], pre_config['label_vec_path'])

if os.path.isfile(vmodel_best_weights_path):
    vmodel.load_state_dict(torch.load(vmodel_best_weights_path))
    print('Loading visual model weights success.')
else:
    print('Weights not found !')
    exit(1)
vmodel.cuda()
vmodel.eval()
# print(vmodel)

# load language model with best weights
lmodel_best_weights_path = train_params['best_model_path']
lmodel = RelationEmbedding(train_params['embedding_dim'] * 2,
                           train_params['embedding_dim'], pre_label_vec_path)
if os.path.isfile(lmodel_best_weights_path):
    lmodel.load_state_dict(torch.load(lmodel_best_weights_path))
    print('Loading language model weights success.')
else:
    print('Weights not found !')
    exit(1)
lmodel.cuda()
lmodel.eval()
# print(lmodel)

# eval
rela_box_label = copy.deepcopy(rela_cands)
img_root = dataset_config.data_config['img_root']
count = 0
for img_id in rela_box_label:
    count += 1
    print('testing [%d/%d]' % (len(rela_box_label.keys()), count))
    box_labels = rela_box_label[img_id]
    if len(box_labels) == 0:
        continue

    # extract cnn feats
    curr_img_boxes = np.array(box_labels)
    img = cv2.imread(os.path.join(img_root, img_id + '.jpg'))

    # pre fc7
    pre_fc7s = ext_cnn_feat(img, curr_img_boxes[:, :4])
    # sbj fc7
    sbj_fc7s = ext_cnn_feat(img, curr_img_boxes[:, 5:9])
    # obj fc7
    obj_fc7s = ext_cnn_feat(img, curr_img_boxes[:, 10:14])

    vfs = np.concatenate((sbj_fc7s, pre_fc7s, obj_fc7s), axis=1)

    for i, box_label in enumerate(rela_box_label[img_id]):

        # extract fc7

        vf = vfs[i]
        vf = vf[np.newaxis, :]
        vf_v = torch.autograd.Variable(torch.from_numpy(vf).float()).cuda()
        pre_lfs_v = torch.autograd.Variable(torch.from_numpy(pre_label_vecs).float()).cuda()
        obj_lfs_v = torch.autograd.Variable(torch.from_numpy(obj_label_vecs).float()).cuda()

        # visual prediction
        v_pre_scores, sbj_scores, obj_scores = vmodel.forward2(vf_v)
        v_pre_scores = v_pre_scores[0]
        sbj_scores = sbj_scores[0]
        obj_scores = obj_scores[0]

        # language prediction
        sbj_ind, sbj_score = gen_prediction(sbj_scores, raw_obj_inds, score_mode)
        sbj_vec = obj_lfs_v[sbj_ind].unsqueeze(0)
        rela_box_label[img_id][i][9] = sbj_ind

        obj_ind, obj_score = gen_prediction(sbj_scores, raw_obj_inds, score_mode)
        obj_vec = obj_lfs_v[obj_ind].unsqueeze(0)
        rela_box_label[img_id][i][14] = obj_ind

        l_pre_scores = lmodel(sbj_vec, obj_vec)[0]

        if use == 'vis_only':
            pre_scores = v_pre_scores
        elif use == 'lan_only':
            pre_scores = l_pre_scores
        elif use == 'vis_lan':
            pre_scores = v_pre_scores * 0.6 + l_pre_scores * 0.4

        pred_pre_ind, pred_pre_score = gen_prediction(pre_scores, raw_pre_inds, score_mode)
        pred_pre_score = pre_scores[pred_pre_ind].cpu().data.numpy().tolist()
        rela_box_label[img_id][i][4] = pred_pre_ind
        rela_box_label[img_id][i].append(pred_pre_score)

output_path = os.path.join(global_config.project_root, 'open_relation', 'output', dataset, 'rela_box_label.bin')
with open(output_path, 'wb') as f:
    pickle.dump(rela_box_label, f)
