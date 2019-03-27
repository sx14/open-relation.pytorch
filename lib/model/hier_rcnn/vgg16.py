# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from lib.model.hier_rcnn.hier_rcnn import _HierRCNN

import pdb

class vgg16(_HierRCNN):
  def __init__(self, objnet, label_vec_path, pretrained=False, class_agnostic=False):
    self.model_path = '../data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _HierRCNN.__init__(self, class_agnostic, objnet, label_vec_path)

  def _init_modules(self):

    # 原始VGG16，注入预训练权重
    vgg = models.vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    # 最后一个线性层不要（分类层）
    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # 最后一个最大池化层不要
    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    # fc7 4096
    self.RCNN_top = vgg.classifier

    # 新建分类层，得分
    # not used in hier-rcnn
    # self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

    # box预测层，box坐标
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)      

  def _head_to_tail(self, pool5):
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)
    return fc7

