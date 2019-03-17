import h5py
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.functional import binary_cross_entropy, sigmoid


def ext_box_feat(rlts):
    # spacial feats
    sbj_boxes = rlts[:, 5:9]
    obj_boxes = rlts[:, 10:14]

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


def loss_test(scores, ys):
    N_pos = 0.0
    N_neg = 0.0
    N_pos_right = 0.0
    N_neg_right = 0.0

    for i in range(ys.shape[0]):
        s = scores[i]
        y = ys[i]
        if y == 0:
            N_neg += 1
            if s >= 0.5:
                N_pos_right += 1
        else:
            N_pos += 1
            if s < 0.5:
                N_neg_right += 1

    recall_pos = N_pos_right / N_pos
    recall_neg = N_neg_right / N_neg
    recall_all = (N_pos_right + N_neg_right) / (N_pos + N_neg)

    loss = binary_cross_entropy(scores, ys)
    return loss, recall_pos, recall_neg, recall_all



class RelaPro(nn.Module):
    def __init__(self, lan_vec_len):
        super(RelaPro, self).__init__()

        self._lan_vec_len = lan_vec_len

        self.spa_stream = nn.Sequential(
            nn.Linear(8, 1),
        )

        self.lan_stream = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(lan_vec_len * 2, lan_vec_len),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(lan_vec_len, 1)
        )

    def forward(self, sbj_vec, obj_vec, rlts):
        sbj_box, obj_box = ext_box_feat(rlts)

        lan_feat = torch.cat([sbj_vec, obj_vec], dim=1)
        box_feat = torch.cat([sbj_box, obj_box], dim=1)

        lan_scores = self.lan_stream(lan_feat)
        box_scores = self.spa_stream(box_feat)
        scores = sigmoid(lan_scores + box_scores)

        ys = rlts[:, 4]  # predicate cls
        ys[ys > 0] = 1    # fg > 0, bg = 0

        loss, recall_pos, recall_neg, recall_all = loss_test(scores, ys)

        return scores, loss, recall_pos, recall_neg, recall_all


