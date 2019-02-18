import os
import pickle
import h5py
import numpy as np
from matplotlib import pyplot as plt
import torch
from open_relation.model.object import model
from open_relation.dataset.dataset_config import DatasetConfig
from open_relation.train.train_config import hyper_params


dataset = 'vrd'

show = 'score'
# show = 'rank'

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



# load model with best weights
best_weights_path = config['latest_weight_path']
net = model.HypernymVisual(config['visual_d'], config['visual_d'], config['embedding_d'], label_vec_path)
if os.path.isfile(best_weights_path):
    net.load_state_dict(torch.load(best_weights_path))
    print('Loading weights success.')
else:
    print('Weights not found !')
    exit(1)
net.cuda()
net.eval()
print(net)

# eval
# simple TF counter
counter = 0
T = 0.0
T1 = 0.0
# expected -> actual
e_p = []
T_ranks = []
F_ranks = []

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
        org_label = objnet.get_node_by_index(org_label_ind)
        scores, _ = net(vf_v)
        scores = scores.cpu().data[0]
        ranked_inds = np.argsort(scores).tolist()
        ranked_inds.reverse()   # descending

        # ====== top ranks ======
        if show == 'rank':
            label_inds = org2path[label2index[org_label]]
            print('\n===== ' + org_label + ' =====')
            print('\n----- answer -----')
            for label_ind in label_inds:
                print(index2label[label_ind])

            preds = ranked_inds[:20]
            print('----- prediction -----')
            for p in preds:
                print('%s : %f' % (index2label[p], scores[p]))
            if counter == 100:
                exit(0)

        # ====== score =====
        else:
            org_indexes = set([label2index[l] for l in org2wn.keys()])
            org_pred_counter = 0
            print('\n===== ' + org_label + ' =====')
            for j, pred in enumerate(ranked_inds):
                if pred in org_indexes:
                    expected = label2index[org_label]

                    if org_pred_counter == 0:
                        e_p.append([expected, pred])

                    if pred == expected:
                        result = 'T: '
                        if org_pred_counter == 0:
                            T += 1
                            T_ranks.append(j+1)
                        elif org_pred_counter == 1:
                            T1 += 1
                    else:
                        result = 'F: '
                        F_ranks.append(j+1)
                    if org_pred_counter < 2:
                        print(result + index2label[pred] + '('+str(j+1)+')')
                    else:
                        break
                    org_pred_counter += 1

print('\n=========================================')
print('accuracy: %.4f' % (T / counter))
print('potential accuracy increment: %.4f' % (T1 / counter))
pickle.dump(e_p, open('e_p.bin', 'wb'))

plt.hist(T_ranks, 100)
plt.xlabel('rank')
plt.ylabel('num')
plt.show()

