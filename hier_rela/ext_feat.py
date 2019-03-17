# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import pprint
import time
import pickle
import random
import numpy as np
import torch

from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list
from lib.model.hier_rela.visual.vgg16 import vgg16 as vgg16_rela
from lib.model.heir_rcnn.vgg16 import vgg16 as vgg16_det
from lib.model.hier_rela.lang.hier_lang import HierLang
from lib.model.hier_rela.hier_rela import HierRela
from lib.model.hier_utils.tree_infer import my_infer
from global_config import PROJECT_ROOT, VG_ROOT, VRD_ROOT
from hier_rela.test_utils import *

from global_config import HierLabelConfig

import pdb

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def extend_neg_samples(im_boxes):
    pos_samples = im_boxes[0]
    neg_samples = pos_samples.clone()

    sbj_boxes = pos_samples[:, 5:10]
    obj_boxes = pos_samples[:, 10:15]

    if pos_samples.shape[0] < 3:
        neg_samples[:, 5:10] = pos_samples[:, 10:15]
        neg_samples[:, 10:15] = pos_samples[:, 5:10]
    else:

        all_boxes = torch.cat((sbj_boxes, obj_boxes), 0)
        all_boxes = np.unique(all_boxes.cpu().data.numpy(), axis=0)

        if all_boxes.shape[0] < neg_samples.shape[0]:
            neg_samples = neg_samples[:all_boxes.shape[0]]
            pos_samples = pos_samples[:all_boxes.shape[0]]

        rand_obj_inds = random.sample(range(all_boxes.shape[0]), neg_samples.shape[0])
        rand_sbj_inds = random.sample(range(all_boxes.shape[0]), neg_samples.shape[0])
        rand_objs = all_boxes[rand_obj_inds]
        rand_sbjs = all_boxes[rand_sbj_inds]

        neg_samples[:, 5:10] = torch.from_numpy(rand_sbjs).float().cuda()
        neg_samples[:, 10:15] = torch.from_numpy(rand_objs).float().cuda()

        neg_pre_xmins = np.min(np.stack((rand_sbjs[:, 0], rand_objs[:, 0]), 1), axis=1)
        neg_pre_ymins = np.min(np.stack((rand_sbjs[:, 1], rand_objs[:, 1]), 1), axis=1)
        neg_pre_xmaxs = np.max(np.stack((rand_sbjs[:, 2], rand_objs[:, 2]), 1), axis=1)
        neg_pre_ymaxs = np.max(np.stack((rand_sbjs[:, 3], rand_objs[:, 3]), 1), axis=1)

        neg_samples[:, 0] = torch.from_numpy(neg_pre_xmins).float().cuda()
        neg_samples[:, 1] = torch.from_numpy(neg_pre_ymins).float().cuda()
        neg_samples[:, 2] = torch.from_numpy(neg_pre_xmaxs).float().cuda()
        neg_samples[:, 3] = torch.from_numpy(neg_pre_ymaxs).float().cuda()

    neg_pre_labels = torch.zeros(neg_pre_xmaxs.shape[0])
    neg_samples[:, 4] = neg_pre_labels

    pos_neg_samples = torch.cat((pos_samples, neg_samples), dim=0)
    pos_neg_samples = pos_neg_samples.unsqueeze(0)
    return pos_neg_samples







def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='vrd', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='../cfgs/vgg16.yml', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true',
                        default=True)
    parser.add_argument('--split', dest='split',
                        help='Do predicate recognition or relationship detection?',
                        default='trainval',
                        # default='test',
                        )
    parser.add_argument('--feat', dest='feat',
                        default='wordbox',
                        # default='wordbox',
                        )

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.dataset == "vg":
        args.imdb_name = "vg_2007_trainval"
        args.imdbval_name = "vg_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        from lib.datasets.vg1000.label_hier.obj_hier import objnet
        from lib.datasets.vg1000.label_hier.pre_hier import prenet
        img_root = os.path.join(VG_ROOT, 'JPEGImages')

    elif args.dataset == "vrd":
        args.imdb_name = "vrd_2016_trainval"
        args.imdbval_name = "vrd_2016_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        from lib.datasets.vrd.label_hier.obj_hier import objnet
        from lib.datasets.vrd.label_hier.pre_hier import prenet
        img_root = os.path.join(VRD_ROOT, 'JPEGImages')

    args.cfg_file = "../cfgs/vgg16.yml"

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    # initilize the network here.
    # Load Detector
    objconf = HierLabelConfig(args.dataset, 'object')
    obj_vec_path = objconf.label_vec_path()
    hierRCNN = vgg16_det(objnet, obj_vec_path, class_agnostic=True)
    hierRCNN.create_architecture()

    preconf = HierLabelConfig(args.dataset, 'predicate')
    pre_vec_path = preconf.label_vec_path()
    hierVis = vgg16_rela(prenet, pre_vec_path, hierRCNN)
    hierVis.create_architecture()

    # Load HierVis
    load_name = '../data/pretrained_model/hier_rela_vis_%s.pth' % args.dataset
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    hierVis.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    # Load HierLan
    hierLan = HierLang(hierRCNN.embedding_len * 2, preconf.label_vec_path())
    load_name = '../data/pretrained_model/hier_rela_lan_%s.pth' % args.dataset
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    hierLan.load_state_dict(checkpoint)

    # get HierRela
    hierRela = HierRela(hierVis, hierLan, objconf.label_vec_path())
    if args.cuda:
        hierRela.cuda()
    hierVis.eval()
    hierLan.eval()
    hierRela.eval()
    print('load model successfully!')

    # Initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    relas_box = torch.FloatTensor(1)
    relas_num = torch.LongTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        relas_num = relas_num.cuda()
        relas_box = relas_box.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    relas_num = Variable(relas_num)
    relas_box = Variable(relas_box)

    if args.cuda:
        cfg.CUDA = True

    # Load gt data
    gt_roidb_path = os.path.join(PROJECT_ROOT, 'hier_rela', 'gt_rela_roidb_%s_%s.bin' % (args.split, args.dataset))
    with open(gt_roidb_path, 'rb') as f:
        gt_roidb = pickle.load(f)
        rela_roidb_use = gt_roidb

    pos_feats = None
    neg_feats = None
    start = time.time()
    N_img = len(rela_roidb_use.keys())
    for i, img_id in enumerate(rela_roidb_use.keys()):
        print('ext [%d/%d]' % (N_img, i+1))
        img_path = os.path.join(img_root, '%s.jpg' % img_id)
        img = cv2.imread(img_path)
        rois_use = rela_roidb_use[img_id]

        # Attention: resized image data
        data = get_input_data(img, rois_use)

        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        relas_box.data.resize_(data[2].size()).copy_(data[2])
        relas_num.data.resize_(data[3].size()).copy_(data[3])

        # extend negative samples
        relas_box = extend_neg_samples(relas_box)
        pos_neg_rela_num = relas_box.size(1)

        with torch.no_grad():
            if args.feat == 'vis':
                feat = hierRela.ext_fc7(im_data, im_info, relas_box, relas_num)
            else:
                feat = hierRela.ext_wordbox(im_data, im_info, relas_box, relas_num)

        pos_rela_num = int(pos_neg_rela_num / 2)
        pos_feat = feat[:pos_rela_num, :].cpu().data.numpy()
        neg_feat = feat[pos_rela_num:, :].cpu().data.numpy()

        if pos_feats is None:
            pos_feats = pos_feat
            neg_feats = neg_feat
        else:
            pos_feats = np.concatenate((pos_feats, pos_feat))
            neg_feats = np.concatenate((neg_feats, neg_feat))

    np.save('%s_%s_pos_feat' % (args.dataset, args.split), pos_feats)
    np.save('%s_%s_neg_feat' % (args.dataset, args.split), neg_feats)


