import os
import pickle
import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from open_relation.model.predicate.model import PredicateVisual
from open_relation.dataset.dataset_config import DatasetConfig
from open_relation.train.train_config import hyper_params
from open_relation.dataset.vrd.label_hier.pre_hier import prenet
from open_relation.dataset.vrd.label_hier.obj_hier import objnet
from open_relation.language.infer.model import RelationEmbedding
from open_relation.language.infer.lang_config import train_params
from open_relation.infer import tree_infer2


def score_pred(pred_ind, raw_label_ind, pred_label, raw_label, pre_net):
    raw2path = pre_net.raw2path()
    if pred_ind == raw_label_ind:
        return 1
    elif pred_ind not in raw2path[raw_label_ind]:
        return 0
    else:
        pre = pre_net.get_node_by_name(raw_label)
        hyper_paths = pre.hyper_paths()
        best_ratio = 0
        for h_path in hyper_paths:
            for i, node in enumerate(h_path):
                if node.name() == pred_label:
                    best_ratio = max((i+1) * 1.0 / (len(h_path)+1), best_ratio)
                    break
        return best_ratio


def top2(ranked_inds, raw_indexes):
    top2_rank = []
    for j, pred in enumerate(ranked_inds):
        if pred in raw_indexes:
            if len(top2_rank) < 2:
                top2_rank.append([pred, j+1])
            else:
                break
    return top2_rank


dataset = 'vrd'

# use = 'vis_only'
# use = 'lan_only'
use = 'vis_lan'

show = 'score'
# show = 'rank'
# show = 'v_l'

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
# simple TF counter
counter = 0
T = 0.0
T1 = 0.0
T_C = 0.0
# expected -> actual
e_p = []
T_ranks = []

visual_feature_root = pre_config['visual_feature_root']
for feature_file_id in test_box_label:
    box_labels = test_box_label[feature_file_id]
    if len(box_labels) == 0:
        continue
    feature_file_name = feature_file_id+'.bin'
    feature_file_path = os.path.join(visual_feature_root, feature_file_name)
    features = pickle.load(open(feature_file_path, 'rb'))
    for i, box_label in enumerate(test_box_label[feature_file_id]):
        counter += 1
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
        ranked_inds.reverse()   # descending

        gt_pre_ind = box_label[4]
        gt_pre_label = prenet.get_node_by_index(gt_pre_ind).name()

        if show == 'rank':
            # ====== top predictions =====
            label_inds = raw2path[label2index[gt_pre_label]]
            print('\n===== ' + gt_pre_label + ' =====')
            print('\n----- answer -----')
            for label_ind in label_inds:
                print(index2label[label_ind])
            preds = ranked_inds[:20]
            print('----- prediction -----')
            for p in preds:
                print('%s : %f' % (index2label[p], v_pre_scores[p]))
            if counter == 100:
                exit(0)
        elif show == 'v_l':
            # ====== vis prediction =====
            # ====== lan prediction =====
            print('\n===== ' + gt_pre_label + ' =====')
            l_top2 = top2(l_ranked_inds, raw_indexes)
            l_pred_ind = l_top2[0][0]
            l_pred_rank = l_top2[0][1]
            l_pred_label = index2label[l_pred_ind]

            if l_pred_ind == gt_pre_ind:
                print('lan >>> T: %s (%d)' % (l_pred_label, l_pred_rank))
            else:
                print('lan >>> F: %s (%d)' % (l_pred_label, l_pred_rank))

            v_top2 = top2(v_ranked_inds, raw_indexes)
            v_pred_ind = v_top2[0][0]
            v_pred_rank = v_top2[0][1]
            v_pred_label = index2label[v_pred_ind]

            if v_pred_ind == gt_pre_ind:
                print('vis >>> T: %s (%d)' % (v_pred_label, v_pred_rank))
            else:
                print('vis >>> F: %s (%d)' % (v_pred_label, v_pred_rank))

            if v_pred_rank > l_pred_rank:
                pred_ind = v_pred_ind
            else:
                pred_ind = l_pred_ind
            if pred_ind == gt_pre_ind:
                T += 1

        else:
            # ====== score ======
            if score_mode == 'raw':
                print('\n===== ' + gt_pre_label + ' =====')
                top2_pred = top2(ranked_inds, raw_indexes)
                for t, (pred_ind, rank) in enumerate(top2_pred):
                    if pred_ind == gt_pre_ind and t == 0:
                        result = 'T: '
                        T += 1
                        T_ranks.append(rank)
                    elif pred_ind == gt_pre_ind and t == 1:
                        result = 'T: '
                        T1 += 1
                    else:
                        result = 'F: '
                    print(result + index2label[pred_ind] + '(' + str(rank) + ')')

            else:
                raw_top2 = top2(ranked_inds, raw_indexes)
                top2_pred = tree_infer2.my_infer(prenet, pre_scores.cpu().data, None, 'pre')
                pred_ind = top2_pred[0][0]
                pred_label = index2label[pred_ind]
                pred_score = score_pred(pred_ind, gt_pre_ind, pred_label, gt_pre_label, prenet)
                T += pred_score
                if pred_score > 0:
                    result = str(counter).ljust(5) + ' T: '
                    T_C += 1
                else:
                    result = str(counter).ljust(5) + ' F: '

                pred_str = (result + gt_pre_label + ' -> ' + pred_label).ljust(40) + ' %.2f | ' % pred_score
                cand_str = ' [%s(%d) , %s(%d)]' % (index2label[raw_top2[0][0]], raw_top2[0][1],
                                                   index2label[raw_top2[1][0]], raw_top2[1][1])
                print(pred_str + cand_str)

            e_p.append([gt_pre_ind, top2_pred[0][0]])


print('\naccuracy: %.4f (%.4f)' % (T / counter, T_C / counter))
print('potential accuracy increment: %.4f' % (T1 / counter))
pickle.dump(e_p, open('e_p.bin', 'wb'))