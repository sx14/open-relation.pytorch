import random
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


class _OrderSimilarity(nn.Module):
    def __init__(self, norm):
        super(_OrderSimilarity, self).__init__()
        self._norm = norm
        self.act = nn.ReLU()

    def forward(self, lab_vecs, vis_vecs):
        order_scores = Variable(torch.zeros(vis_vecs.size()[0], lab_vecs.size()[0])).cuda()
        for i in range(vis_vecs.size()[0]):
            # hyper - hypo
            sub = lab_vecs - vis_vecs[i]
            # max(sub, 0)
            sub = self.act(sub)
            # norm 2
            order_dis = sub.norm(p=self._norm, dim=1)
            order_sim = -order_dis
            order_scores[i] = order_sim
        return order_scores


class _HierRCNN(nn.Module):
    """ Hier RCNN """
    def __init__(self, class_agnostic, objnet, label_vec_path):
        super(_HierRCNN, self).__init__()

        # config heir classes
        # all classes
        self.classes = objnet.get_all_labels()
        self.n_classes = len(self.classes)
        self.class_agnostic = class_agnostic
        # min negative label num
        self.n_neg_classes = objnet.neg_num()
        self.objnet = objnet

        # label vectors
        with h5py.File(label_vec_path, 'r') as f:
            label_vecs = np.array(f['label_vec'])
            self.label_vecs = Variable(torch.from_numpy(label_vecs).float()).cuda()

        # visual embedding vector length
        self.embedding_len = self.label_vecs.size(1)

        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        # fc7(4096) -> emb(600)
        self.order_embedding = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, self.embedding_len))

        self.order_score = _OrderSimilarity(norm=2)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, use_rpn=True):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        if use_rpn:
            # feed base feature map tp RPN to obtain rois
            rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

            # if it is training phrase, then use ground trubut bboxes for refining
            if self.training:
                roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
                rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

                rois_label = Variable(rois_label.view(-1).long())
                rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
                rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
                rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
            else:
                rois_label = None
                rois_target = None
                rois_inside_ws = None
                rois_outside_ws = None
                rpn_loss_cls = 0
                rpn_loss_bbox = 0

            rois = Variable(rois)
            # do roi pooling based on predicted rois
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            raw_rois = torch.zeros(gt_boxes.size())
            raw_rois[0, :, 1:] = gt_boxes[0, :, :4]
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


        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # ===== order embedding here =====\
        # visual embedding
        vis_embedding = self.order_embedding(pooled_feat)
        # compute order similarity for hier labels
        cls_score = self.order_score(self.label_vecs, vis_embedding)
        # ===== order embedding here =====/

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            pos_negs = self._loss_labels(rois_label)
            loss_score, y = self._prepare_loss_input(cls_score, pos_negs)

            # classification loss
            RCNN_loss_cls = F.cross_entropy(loss_score, y)
            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_score = cls_score.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_score, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    # prepare cls scores [pos, neg1, neg2, ...]
    def _prepare_loss_input(self, cls_scores, pos_negs):
        loss_scores = Variable(torch.zeros(len(cls_scores), len(pos_negs[0]))).float().cuda()
        obj_count = 0
        for i in range(len(cls_scores)):
            if pos_negs[i][0] == 0:
                # abort background
                continue
            scores = cls_scores[i]
            loss_scores[obj_count] = scores[pos_negs[i]]
            obj_count += 1

        loss_scores = loss_scores[:obj_count]
        y = Variable(torch.zeros(len(loss_scores))).long().cuda()
        return loss_scores, y

    # prepare labels [pos, neg1, neg2, ...]
    def _loss_labels(self, rois_label):
        loss_labels = np.zeros((rois_label.size()[0], 1+self.n_neg_classes)).astype(np.int)
        for i, gt_ind in enumerate(rois_label):
            gt_label = self.objnet.get_all_labels()[gt_ind]
            gt_node = self.objnet.get_node_by_name(gt_label)
            all_pos_inds = set(gt_node.trans_hyper_inds())
            all_neg_inds = list(set(range(self.objnet.label_sum())) - all_pos_inds)
            loss_labels[i] = [rois_label[i]] + random.sample(all_neg_inds, self.n_neg_classes)
        return loss_labels

    # extract roi fc7
    def ext_feat(self, im_data, im_info, gt_boxes, num_boxes, use_rpn=True):
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        if use_rpn:
            # feed base feature map tp RPN to obtain rois
            rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
            rois = Variable(rois)
            # do roi pooling based on predicted rois
        else:
            raw_rois = torch.zeros(gt_boxes.size())
            raw_rois[0, :, 1:] = gt_boxes[0, :, :4]
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

        # feed pooled features to top model
        fc7 = self._head_to_tail(pooled_feat)

        return fc7


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

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(list(self.order_embedding._modules.values())[0], 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(list(self.order_embedding._modules.values())[-1], 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
