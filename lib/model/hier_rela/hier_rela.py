import h5py
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable


class HierRela(nn.Module):
    def __init__(self, hierVis=None, hierLang=None, obj_vec_path=None):
        super(HierRela, self).__init__()

        if hierVis is None and hierLang is None:
            print('HierRela: Pass HierVis or HierLang or both.')
            exit(-1)

        self._hierVis = hierVis
        self._hierLang = hierLang

        # label vectors
        with h5py.File(obj_vec_path, 'r') as f:
            obj_lab_vecs = np.array(f['label_vec'])

        with torch.no_grad():
            self._obj_lab_vecs = Variable(torch.from_numpy(obj_lab_vecs).float()).cuda()

    def forward(self, im_data, im_info, gt_relas, num_relas):
        batch_size = im_data.size(0)

        pre_label = gt_relas[:, :, 4][0].long()
        sbj_label = gt_relas[:, :, 9][0].long()
        obj_label = gt_relas[:, :, 14][0].long()

        if self._hierVis is None and self._hierLang is None:
            print('HierRela: no visual module, no language module.')
            exit(-1)

        if self._hierVis is not None:
            _, vis_score, _, _ = self._hierVis(im_data, im_info, gt_relas, num_relas)
            vis_score = vis_score[0]

        if self._hierLang is not None:
            sbj_lab_vecs = self._obj_lab_vecs[sbj_label]
            obj_lab_vecs = self._obj_lab_vecs[obj_label]
            lan_score = self._hierLang(sbj_lab_vecs, obj_lab_vecs)

        if self._hierLang is None:
            lan_score = vis_score

        if self._hierVis is None:
            vis_score = lan_score

        score = 0.7 * lan_score + 0.3 * vis_score
        score[score < -3] = -3

        pre_boxes = gt_relas[:, :, :5]
        raw_pre_rois = torch.zeros(pre_boxes.size())
        raw_pre_rois[:, :, 1:] = pre_boxes[:, :, :4]
        rois = Variable(raw_pre_rois).cuda()

        score = score.view(batch_size, rois.size(1), -1)
        vis_score = vis_score.view(batch_size, rois.size(1), -1)
        lan_score = lan_score.view(batch_size, rois.size(1), -1)
        rois_label = torch.stack((pre_label, sbj_label, obj_label), dim=1)
        rois_label.unsqueeze(0)

        cls_score = 0

        return rois, score, cls_score, rois_label, vis_score, lan_score






