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
import torchvision.models as models
from lib.model.hier_rela.visual.hier_rela_vis import _HierRelaVis


class vgg16(_HierRelaVis):
  def __init__(self, objnet, level_vec_path, hierRCNN, pretrained=False):
    self.model_path = '../data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained

    _HierRelaVis.__init__(self, objnet, level_vec_path, hierRCNN)

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

    # fix RCNN_base
    for p in self.RCNN_base.parameters(): p.requires_grad = False
    self.RCNN_base.eval()

    # Fix the layers before conv3:
    #for layer in range(10):
    #  for p in self.RCNN_base[layer].parameters(): p.requires_grad = False


    # fc7 4096
    self.RCNN_top = vgg.classifier
    for p in self.RCNN_top.parameters(): p.requires_grad = False
    self.RCNN_top.eval()


  def _head_to_tail(self, pool5):
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)
    return fc7

