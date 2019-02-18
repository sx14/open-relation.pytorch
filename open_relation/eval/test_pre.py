import os
import copy
import pickle
import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from open_relation.global_config import project_root
from open_relation.model.predicate.model import PredicateVisual
from open_relation.dataset.dataset_config import DatasetConfig
from open_relation.train.train_config import hyper_params
from open_relation.dataset.vrd.label_hier.pre_hier import prenet
from open_relation.dataset.vrd.label_hier.obj_hier import objnet
from open_relation.language.infer.model import RelationEmbedding
from open_relation.language.infer.lang_config import train_params
from open_relation.infer import tree_infer2



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
test_list_path = os.path.join(dataset_config.extra_config['predicate'].prepare_root, 'test_box_label.bin')
test_box_label = pickle.load(open(test_list_path))

pre_label_vec_path = pre_config['label_vec_path']
label_embedding_file = h5py.File(pre_label_vec_path, 'r')
pre_label_vecs = np.array(label_embedding_file['label_vec'])
label_embedding_file.close()

obj_label_vec_path = obj_config['label_vec_path']
label_embedding_file = h5py.File(obj_label_vec_path, 'r')
obj_label_vecs = np.array(label_embedding_file['label_vec'])
label_embedding_file.close()

# prepare label maps
raw2path = prenet.raw2path()
label2index = prenet.label2index()
index2label = prenet.index2label()
raw_indexes = set([label2index[l] for l in prenet.get_raw_labels()])

# load visual model with best weights
vmodel_best_weights_path = pre_config['best_weight_path']
vmodel = PredicateVisual(obj_config['visual_d'], obj_config['hidden_d'],
                         obj_config['embedding_d'], obj_config['label_vec_path'], obj_config['best_weight_path'],
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
lmodel = RelationEmbedding(train_params['embedding_dim'] * 2, train_params['embedding_dim'], pre_label_vec_path)
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
pred_box_label = copy.deepcopy(test_box_label)
visual_feature_root = pre_config['visual_feature_root']
count = 0
for img_id in test_box_label:
    count += 1
    print('testing [%d/%d]' % (len(test_box_label.keys()), count))
    box_labels = test_box_label[img_id]
    if len(box_labels) == 0:
        continue


    feature_file_name = img_id + '.bin'
    feature_file_path = os.path.join(visual_feature_root, feature_file_name)
    features = pickle.load(open(feature_file_path, 'rb'))


    for i, box_label in enumerate(test_box_label[img_id]):
        vf = features[i]
        vf = vf[np.newaxis, :]
        vf_v = torch.autograd.Variable(torch.from_numpy(vf).float()).cuda()
        pre_lfs_v = torch.autograd.Variable(torch.from_numpy(pre_label_vecs).float()).cuda()
        obj_lfs_v = torch.autograd.Variable(torch.from_numpy(obj_label_vecs).float()).cuda()

        # visual prediction
        v_pre_scores, _ = vmodel(vf_v)
        v_pre_scores = v_pre_scores[0]
        v_ranked_inds = np.argsort(v_pre_scores.cpu().data).tolist()  # ascend
        v_ranked_inds.reverse()  # descending

        # language prediction
        sbj_ind = box_label[9]
        sbj_vec = obj_lfs_v[sbj_ind].unsqueeze(0)

        obj_ind = box_label[14]
        obj_vec = obj_lfs_v[obj_ind].unsqueeze(0)

        l_pre_scores = lmodel(sbj_vec, obj_vec)[0]
        l_ranked_inds = np.argsort(l_pre_scores.cpu().data).tolist()
        l_ranked_inds.reverse()  # descending

        if use == 'vis_only':
            pre_scores = v_pre_scores
        elif use == 'lan_only':
            pre_scores = l_pre_scores
        elif use == 'vis_lan':
            pre_scores = v_pre_scores * 0.6 + l_pre_scores * 0.4

        ranked_inds = np.argsort(pre_scores.cpu().data).tolist()
        ranked_inds.reverse()  # descending

        # ====== score ======
        if score_mode == 'raw':
            for ind in ranked_inds:
                if ind in raw_indexes:
                    pred_pre_ind = ind
                    pred_pre_score = pre_scores[pred_pre_ind].cpu().data.numpy().tolist()
                    pred_box_label[img_id][i][4] = pred_pre_ind
                    pred_box_label[img_id][i].append(pred_pre_score)
                    break


output_path = os.path.join(project_root, 'open_relation', 'output', dataset,'pre_box_label.bin')
with open(output_path, 'wb') as f:
    pickle.dump(pred_box_label, f)


