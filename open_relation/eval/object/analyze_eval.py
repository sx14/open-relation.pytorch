import pickle
import numpy as np

dataset = 'vrd'
if dataset == 'vrd':
    from open_relation.dataset.vrd.label_hier.obj_hier import objnet
else:
    from open_relation.dataset.vg.label_hier.obj_hier import objnet


# prepare label maps
org2wn = objnet.raw2wn()
label2index = objnet.label2index()
index2label = objnet.index2label()


# evaluation result
e_p_path = 'e_p.bin'
e_p_list = pickle.load(open(e_p_path, 'rb'))

# cal acc for each class
e_freq = dict()
e_acc = dict()

# prediction matrix : expected -> prediction
mat2org = [label2index[i] for i in org2wn.keys()]
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

show_item = 'road'
print('\n======== %s ========' % show_item)
preds = e_p_mat[org2mat[label2index[show_item]], :]
ranked_inds = np.argsort(preds)
for i in range(len(ranked_inds) -1, -1, -1):
    label_str = index2label[mat2org[ranked_inds[i]]].ljust(15)
    pred_count = preds[ranked_inds[i]]
    if pred_count > 0:
        print(label_str + ': %d' % pred_count)

