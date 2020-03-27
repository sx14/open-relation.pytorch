import pickle
import numpy as np

dataset = 'vg'
top_k = 100
pred_roidb_path = '../hier_rela/rela_box_label_%s_hier.bin' % (dataset)
pred_roidb = pickle.load(open(pred_roidb_path))
print('image db loaded')

for img_id in pred_roidb:
    pr_curr = pred_roidb[img_id]
    pr_curr = np.array(pr_curr)
    pr_labels = []
    obj_labels = []
    sbj_labels = []
    _, uni_idx = np.unique(pr_curr[:, [4, 9, 14]], axis=0, return_index=True)
    pr_curr = pr_curr[uni_idx]
    for i in range(pr_curr.shape[0]):
        pred_score = pr_curr[i, 15]
        if pred_score >= 30:
            pred_score -= 30
        elif pred_score >= 20:
            pred_score -= 20
        elif pred_score >= 10:
            pred_score -= 10
        pr_curr[i, 15] = pred_score

    pr_curr = pr_curr[pr_curr[:, 15].argsort()[::-1]]
    pred_roidb[img_id] = pr_curr[:top_k, [4,9,14,15]]
with open('../hier_rela/rela_box_label_%s_hier_pure.bin' % (dataset), 'wb') as f:
    pickle.dump(pred_roidb, f)
