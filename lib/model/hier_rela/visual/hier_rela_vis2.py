import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import h5py
from lib.model.utils.config import cfg
from lib.model.rpn.rpn import _RPN
from lib.model.roi_pooling.modules.roi_pool import _RoIPooling
from lib.model.roi_crop.modules.roi_crop import _RoICrop
from lib.model.roi_align.modules.roi_align import RoIAlignAvg
from lib.model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from lib.model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from lib.model.hier_utils.hier_utils import OrderSimilarity, OrderLoss


class _HierRelaVis(nn.Module):
    """ Hier RCNN """
    def __init__(self, prenet, pre_label_vec_path, obj_label_vec_path):
        super(_HierRelaVis, self).__init__()

        # object detector
        self._loss = OrderLoss(prenet)
        # config heir classes
        # all classes
        self.classes = prenet.get_all_labels()
        self.n_classes = len(self.classes)
        # min negative label num

        # label vectors
        with h5py.File(pre_label_vec_path, 'r') as f:
            pre_label_vecs = np.array(f['label_vec'])

        with torch.no_grad():
            self.pre_label_vecs = Variable(torch.from_numpy(pre_label_vecs).float()).cuda()

        with h5py.File(obj_label_vec_path, 'r') as f:
            obj_label_vecs = np.array(f['label_vec'])

        with torch.no_grad():
            self.obj_label_vecs = Variable(torch.from_numpy(obj_label_vecs).float()).cuda()

        # visual embedding vector length
        self.embedding_len = self.pre_label_vecs.size(1)
        self.obj_embedding_len = self.obj_label_vecs.size(1)

        # loss
        self.RCNN_loss_cls = 0
        # self.RCNN_loss_bbox = 0

        # define rpn
        # self.RCNN_rpn = _RPN(self.dout_base_model)
        # self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        # Our model
        self.vis_embedding = nn.Sequential(
            nn.Linear(4096 * 3, 4096 * 3),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096 * 3, self.obj_embedding_len))

        self.order_embedding = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.obj_embedding_len * 3 + 8, self.obj_embedding_len * 3 + 8),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.obj_embedding_len * 3, self.embedding_len))

        self.order_score = OrderSimilarity(norm=2)

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        if num_boxes.item() > 0:
            gt_boxes = gt_boxes[:, :num_boxes.item(), :]
        else:
            print('[sunx] Attention: No rela box in current batch.')

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        pre_label = gt_boxes[:, :, 4][0]
        sbj_label = gt_boxes[:, :, 9][0]
        obj_label = gt_boxes[:, :, 14][0]

        pre_boxes = gt_boxes[:, :, :5]
        sbj_boxes = gt_boxes[:, :, 5:10]
        obj_boxes = gt_boxes[:, :, 10:15]

        raw_rois = torch.zeros(pre_boxes.shape[0], pre_boxes.shape[1]*3, pre_boxes.shape[2])
        raw_rois[:, :pre_boxes.shape[1], 1:] = pre_boxes[:, :, :4]
        raw_rois[:, pre_boxes.shape[1]:pre_boxes.shape[1]*2, 1:] = sbj_boxes[:, :, :4]
        raw_rois[:, pre_boxes.shape[1]*2:pre_boxes.shape[1]*3, 1:] = obj_boxes[:, :, :4]
        rois = Variable(raw_rois).cuda()

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model(fc7)
        pooled_feat = self._head_to_tail(pooled_feat)

        pre_pooled_feat = pooled_feat[:pre_boxes.shape[1], :]
        sbj_pooled_feat = pooled_feat[pre_boxes.shape[1]:pre_boxes.shape[1]*2, :]
        obj_pooled_feat = pooled_feat[pre_boxes.shape[1]*2:pre_boxes.shape[1]*3, :]

        vis_feat_use = torch.cat([sbj_pooled_feat, pre_pooled_feat, obj_pooled_feat], 1)
        vis_embedding = self.vis_embedding(vis_feat_use)

        spacial_feat = self._ext_box_feat(gt_boxes)
        sbj_spacial_feat = spacial_feat[:, :4]
        obj_spacial_feat = spacial_feat[:, 4:]

        sbj_vecs = self.obj_label_vecs[sbj_label.view(-1).long(), :]
        obj_vecs = self.obj_label_vecs[obj_label.view(-1).long(), :]

        feat_use = torch.cat([sbj_vecs, sbj_spacial_feat, obj_vecs, obj_spacial_feat, vis_embedding], 1)
        pre_embedding = self.order_embedding(feat_use)

        # compute order similarity
        if pre_embedding.size(0) < 30:
            # fast, memory consuming
            cls_score_use = self.order_score.forward(self.pre_label_vecs, pre_embedding)
        else:
            # slow, memory saving
            cls_score_use = self.order_score.forward1(self.pre_label_vecs, pre_embedding)
        # ===== order embedding here =====/

        RCNN_loss_cls = 0

        if self.training:
            RCNN_loss_cls = self._loss.forward(cls_score_use, pre_label)

        cls_score = cls_score_use.view(batch_size, rois.size(1), -1)
        rois_label = torch.stack((pre_label, sbj_label, obj_label), dim=1)
        rois_label.unsqueeze(0)

        return rois, cls_score, RCNN_loss_cls, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(list(self.pre_hidden._modules.values())[0], 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(list(self.sbj_hidden._modules.values())[0], 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(list(self.obj_hidden._modules.values())[0], 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(list(self.vis_embedding._modules.values())[2], 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(list(self.vis_embedding._modules.values())[-1], 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(list(self.order_embedding._modules.values())[2], 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(list(self.order_embedding._modules.values())[-1], 0, 0.01, cfg.TRAIN.TRUNCATED)


    def _ext_box_feat(self, gt_relas):

        # spacial feats
        sbj_boxes = gt_relas[0, :, 5:9]
        obj_boxes = gt_relas[0, :, 10:14]

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

        # transe
        # spatial_vecs = sbj_feat_vecs - obj_feat_vecs
        spatial_vecs = torch.cat([sbj_feat_vecs, obj_feat_vecs], dim=1)
        return spatial_vecs


    def create_architecture(self):
        self._init_modules()
        self._init_weights()
