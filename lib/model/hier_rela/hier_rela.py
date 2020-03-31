import h5py
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable


class HierRela(nn.Module):
    def __init__(self, hierVis=None, hierSpatial=None):
        super(HierRela, self).__init__()

        if hierVis is None and hierSpatial is None:
            print('HierRela: Pass HierVis or HierSpatial or both.')
            exit(-1)

        self._hierVis = hierVis
        self._hierSpatial = hierSpatial

    def forward(self, im_data, im_info, gt_relas, spa_maps, num_relas):
        batch_size = im_data.size(0)

        pre_label = gt_relas[:, :, 4][0].long()
        sbj_label = gt_relas[:, :, 9][0].long()
        obj_label = gt_relas[:, :, 14][0].long()

        if self._hierVis is None and self._hierSpatial is None:
            print('HierRela: no visual module, no spatial module.')
            exit(-1)

        if self._hierVis is not None:
            _, vis_score, _, _ = self._hierVis(im_data, im_info, gt_relas, num_relas)
            vis_score = vis_score[0]

        if self._hierSpatial is not None:
            spa_score = self._hierSpatial(spa_maps[0], pre_label)

        if self._hierSpatial is None:
            spa_score = vis_score

        if self._hierVis is None:
            vis_score = spa_score

        score = 0.7 * spa_score + 0.3 * vis_score
        score[score < -3] = -3

        pre_boxes = gt_relas[:, :, :5]
        raw_pre_rois = torch.zeros(pre_boxes.size())
        raw_pre_rois[:, :, 1:] = pre_boxes[:, :, :4]
        rois = Variable(raw_pre_rois).cuda()

        score = score.view(batch_size, rois.size(1), -1)
        vis_score = vis_score.view(batch_size, rois.size(1), -1)
        spa_score = spa_score.view(batch_size, rois.size(1), -1)
        rois_label = torch.stack((pre_label, sbj_label, obj_label), dim=1)
        rois_label.unsqueeze(0)

        cls_score = 0

        return rois, score, cls_score, rois_label, vis_score, spa_score






