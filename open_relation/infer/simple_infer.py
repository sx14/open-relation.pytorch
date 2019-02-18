# -*- coding: utf-8 -*-
import numpy as np



def simple_infer(scores, org2path, label2index):
    # all original label indexes
    org_label_inds = org2path.keys()

    # find the top 2 original label predictions
    ranked_inds = np.argsort(scores).tolist()
    ranked_inds.reverse()           # descending
    ranks = [0] * len(label2index.keys())   # label_ind 2 rank
    top_org_ind_ranks = []
    iter = 0
    while len(top_org_ind_ranks) < 2:
        if ranked_inds[iter] in org_label_inds:
            top_org_ind_ranks.append([ranked_inds[iter], iter+1])
        ranks[ranked_inds[iter]] = iter + 1
        iter += 1

    # collect scores of labels on the top 2 paths
    pred_paths = [org2path[org_ind_rank[0]] for org_ind_rank in top_org_ind_ranks]
    pred_path_ranks = [[0] * len(org2path[org_ind_rank[0]]) for org_ind_rank in top_org_ind_ranks]
    for p, path in enumerate(pred_paths):
        for i, l in enumerate(path):
            pred_path_ranks[p][i] = ranks[l]

    # default predication
    final_pred = pred_paths[0][-1]

    top_org_rank_diff = top_org_ind_ranks[1][1] - top_org_ind_ranks[0][1]

    if top_org_ind_ranks[0][1] < 40:
        # 正确的几率很高，除非top1与top2非常相似
        diff_interval = 40
        overlap_ratio_thr = 0.85
    elif top_org_ind_ranks[0][1] < 150:
        diff_interval = 60
        overlap_ratio_thr = 0.4
    else:
        # 正确的几率很低
        diff_interval = 150
        overlap_ratio_thr = 0.2

    if top_org_rank_diff < diff_interval:
        # top1 is close to top2
        # cal top1 and top2 overlap
        overlap = set(pred_paths[0]) & set(pred_paths[1])
        overlap_ratio = len(overlap) * 1.0 / min(len(pred_paths[0]), len(pred_paths[1]))

        # overlap is large
        if overlap_ratio > overlap_ratio_thr:
            final_pred = max(overlap)

    return final_pred, top_org_ind_ranks











