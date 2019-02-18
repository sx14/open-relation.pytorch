import os
import pickle
import h5py
import numpy as np
import torch
from nltk.corpus import wordnet as wn
from open_relation.infer import tree_infer2
from open_relation.model.object import model
from open_relation.dataset.dataset_config import DatasetConfig
from open_relation.train.train_config import hyper_params


def score_pred(pred_ind, org_label_ind, pred_label, wn_label, org2path):
    if pred_ind == org_label_ind:
        return 1
    elif pred_ind not in org2path[org_label_ind]:
        return 0
    else:
        wn_node = wn.synset(wn_label)
        hyper_paths = wn_node.hypernym_paths()
        best_ratio = 0
        for h_path in hyper_paths:
            for i, node in enumerate(h_path):
                if node.name() == pred_label:
                    best_ratio = max((i+1) * 1.0 / (len(h_path)+1), best_ratio)
                    break
        return best_ratio


dataset = 'vrd'
dataset_config = DatasetConfig(dataset)

if dataset == 'vrd':
    from open_relation.dataset.vrd.label_hier.obj_hier import objnet
else:
    from open_relation.dataset.vg.label_hier.obj_hier import objnet

# prepare feature
config = hyper_params[dataset]['object']
test_list_path = os.path.join(dataset_config.extra_config['object'].prepare_root, 'test_box_label.bin')
test_box_label = pickle.load(open(test_list_path))
label_vec_path = config['label_vec_path']
label_embedding_file = h5py.File(label_vec_path, 'r')
label_vecs = np.array(label_embedding_file['label_vec'])

# prepare label maps

org2wn = objnet.raw2wn()
org2path = objnet.raw2path()
label2index = objnet.label2index()
index2label = objnet.index2label()
org_indexes = [label2index[i] for i in org2wn.keys()]

# load model with best weights
best_weights_path = config['latest_weight_path']
net = model.HypernymVisual(config['visual_d'], config['hidden_d'],
                           config['embedding_d'], label_vec_path)
if os.path.isfile(best_weights_path):
    net.load_state_dict(torch.load(best_weights_path))
    print('Loading weights success.')
net.cuda()
net.eval()
print(net)

# eval
# simple TF counter
counter = 0
T = 0.0
T_C = 0.0
# expected -> actual
e_p = []

# rank_scores = tree_infer2.cal_rank_scores1(len(index2label))
visual_feature_root = config['visual_feature_root']
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
        lfs_v = torch.autograd.Variable(torch.from_numpy(label_vecs).float()).cuda()
        org_label_ind = box_label[4]
        org_label = objnet.get_node_by_index(org_label_ind).name()
        scores, _ = net(vf_v)
        scores = scores.cpu().data[0]
        top2pred = tree_infer2.my_infer(objnet, scores, None, 'obj')
        pred_ind = top2pred[0][0]
        # pred_ind, cands = tree_infer.my_infer(scores, org2path, org2pw, label2index, index2label, rank_scores)
        # pred_ind, cands = simple_infer.simple_infer(scores, org2path, label2index)
        pred_score = score_pred(pred_ind, org_label_ind, index2label[pred_ind], org2wn[org_label][0], org2path)
        T += pred_score
        if pred_score > 0:
            T_C += 1
            result = str(counter).ljust(5) + ' T: '
        else:
            result = str(counter).ljust(5) + ' F: '

        pred_str = (result + org_label + ' -> ' + index2label[pred_ind]).ljust(40) + ' %.2f | ' % pred_score
        cand_str = ' [%s(%d) , %s(%d)]' % (index2label[top2pred[0][0]], top2pred[0][1], index2label[top2pred[1][0]], top2pred[1][1])
        print(pred_str + cand_str)

print('\n=========================================')
print('accuracy: %.4f (%.4f)' % (T / counter, T_C / counter))


