import pickle
import numpy as np
from open_relation.dataset import dataset_config
from open_relation.dataset.vrd.label_hier.pre_hier import prenet
# from open_relation1.dataset.vrd.predicate.pre_hier import PreNet


# prepare label maps
label2index_path = dataset_config.vrd_predicate_config['label2index_path']
label2index = pickle.load(open(label2index_path))
index2label_path = dataset_config.vrd_predicate_config['index2label_path']
index2label = pickle.load(open(index2label_path))


# evaluation result
e_p_path = 'e_p.bin'
e_p_list = pickle.load(open(e_p_path, 'rb'))

# cal acc for each class
e_freq = dict()
e_acc = dict()

# prediction matrix : expected -> prediction
mat2org = [label2index[i] for i in prenet.get_raw_labels()]
org2mat = dict()
for i, org in enumerate(mat2org):
    org2mat[org] = i
e_p_mat = np.zeros((len(mat2org), len(mat2org)))


for e_p in e_p_list:
    if e_p[0] not in e_acc:
        e_acc[e_p[0]] = 0
        e_freq[e_p[0]] = 0
    e_freq[e_p[0]] += 1
    if e_p[0] == e_p[1]:
        e_acc[e_p[0]] += 1
    e_p_mat[org2mat[e_p[0]], org2mat[e_p[1]]] += 1

for e in e_acc:
    e_acc[e] = e_acc[e] * 1.0 / e_freq[e]

sorted_freq = sorted(e_freq.items(), lambda x, y: cmp(y[1], x[1]))
for item in sorted_freq:
    label_str = index2label[item[0]].ljust(15)
    print(label_str + ': %.2f (%.3f)' % (e_acc[item[0]], item[1] * 1.0 / len(e_p_list)))

show_item = 'street'
print('\n======== %s ========' % show_item)
preds = e_p_mat[org2mat[label2index[show_item]], :]
ranked_inds = np.argsort(preds)
for i in range(len(ranked_inds) -1, -1, -1):
    label_str = index2label[mat2org[ranked_inds[i]]].ljust(15)
    pred_count = preds[ranked_inds[i]]
    if pred_count > 0:
        print(label_str + ': %d' % pred_count)

