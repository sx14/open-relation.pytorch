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
    def __init__(self, prenet, label_vec_path, hierRCNN):
        super(_HierRelaVis, self).__init__()

        # object detector
        self._hierRCNN = hierRCNN
        self._loss = OrderLoss(prenet)
        # config heir classes
        # all classes
        self.classes = prenet.get_all_labels()
        self.n_classes = len(self.classes)
        # min negative label num

        # label vectors
        with h5py.File(label_vec_path, 'r') as f:
            label_vecs = np.array(f['label_vec'])

        with torch.no_grad():
            self.label_vecs = Variable(torch.from_numpy(label_vecs).float()).cuda()

        # visual embedding vector length
        self.embedding_len = self.label_vecs.size(1)

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
        # fc7(4096) -> emb(600)
        self.order_in_embedding = nn.Sequential(
            nn.Linear(4096*3, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, self.embedding_len))

        # self.order_ex_embedding = nn.Sequential(
        #     nn.Linear(self.embedding_len + self._hierRCNN.embedding_len * 2,
        #               self.embedding_len + self._hierRCNN.embedding_len * 2),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(self.embedding_len + 2 * self._hierRCNN.embedding_len,
        #               self.embedding_len))

        self.order_score = OrderSimilarity(norm=2)

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        pre_label = gt_boxes[:, :, 4][0]
        sbj_label = gt_boxes[:, :, 9][0]
        obj_label = gt_boxes[:, :, 14][0]

        pre_boxes = gt_boxes[:, :, :5]
        sbj_boxes = gt_boxes[:, :, 5:10]
        obj_boxes = gt_boxes[:, :, 10:15]

        raw_pre_rois = torch.zeros(pre_boxes.size())
        raw_pre_rois[:, :, 1:] = pre_boxes[:, :, :4]
        rois = Variable(raw_pre_rois).cuda()

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pre_pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pre_pooled_feat = F.max_pool2d(pre_pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pre_pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pre_pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model(fc7)
        pre_pooled_feat = self._head_to_tail(pre_pooled_feat)

        sbj_obj_boxes = torch.cat([sbj_boxes, obj_boxes], 1)

        with torch.no_grad():
            sbj_obj_pooled_feat, \
            sbj_obj_embedding = self._hierRCNN.ext_feat(im_data, im_info,
                                                        sbj_obj_boxes,num_boxes * 2,
                                                        use_rpn=False)

        num_boxes_padding = gt_boxes.size(1)
        sbj_pooled_feat = sbj_obj_pooled_feat[:num_boxes_padding, :]
        obj_pooled_feat = sbj_obj_pooled_feat[num_boxes_padding:, :]
        sbj_embedding = sbj_obj_embedding[:num_boxes_padding, :]
        obj_embedding = sbj_obj_embedding[num_boxes_padding:, :]

        # ===== class prediction part =====
        # ===== order embedding here =====\
        pooled_feat_use = torch.cat([sbj_pooled_feat, pre_pooled_feat, obj_pooled_feat], 1)

        # visual embedding
        #pre_embedding0 = self.order_in_embedding(pooled_feat_use)
        # pre_feat = torch.cat([sbj_embedding, pre_embedding0, obj_embedding], 1)
        # pre_embedding = self.order_ex_embedding(pre_feat)

        pre_embedding = self.order_in_embedding(pooled_feat_use)


        # compute order similarity
        if pre_embedding.size(0) < 30:
            # fast, memory consuming
            cls_score_use = self.order_score.forward(self.label_vecs, pre_embedding)
        else:
            # slow, memory saving
            cls_score_use = self.order_score.forward1(self.label_vecs, pre_embedding)
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

        normal_init(list(self.order_in_embedding._modules.values())[0], 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(list(self.order_ex_embedding._modules.values())[-1], 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
