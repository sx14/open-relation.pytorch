# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import pprint
import time
import cv2
import torch
from torch.autograd import Variable
import pickle
from lib.roi_data_layer.roidb import combined_roidb
from lib.roi_data_layer.roibatchLoader import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.nms.nms_wrapper import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv
from lib.model.utils.net_utils import vis_detections
from lib.model.hier_rcnn.vgg16 import vgg16
from lib.model.hier_rcnn.resnet import resnet
from lib.model.hier_utils.infer_tree import InferTree
from global_config import HierLabelConfig, PROJECT_ROOT
from hier_det.test_utils import det_recall, load_vrd_det_boxes

import pdb

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


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
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='vgg16', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="hier_output",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true',
                        default=True)
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large image scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=20, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=7547, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)

    if args.dataset == "vg":
        args.imdb_name = "vg_2007_trainval"
        args.imdbval_name = "vg_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        from lib.datasets.vg200.label_hier.obj_hier import objnet
        args.class_agnostic = True

    elif args.dataset == "vrd":
        args.imdb_name = "vrd_2007_trainval"
        args.imdbval_name = "vrd_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        from lib.datasets.vrd.label_hier.obj_hier import objnet
        args.class_agnostic = True

    args.cfg_file = "../cfgs/{}_ls.yml".format(args.net) if args.large_scale else "../cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'hier_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    # initilize the network here.
    if args.net == 'vgg16':
        labelconf = HierLabelConfig(args.dataset, 'object')
        label_vec_path = labelconf.label_vec_path()
        hierRCNN = vgg16(objnet, label_vec_path, pretrained=False, class_agnostic=True)
    elif args.net == 'res101':
        labelconf = HierLabelConfig(args.dataset, 'object')
        label_vec_path = labelconf.label_vec_path()
        hierRCNN = resnet(objnet, label_vec_path, pretrained=False, class_agnostic=True)
    else:
        print("network is not defined")
        pdb.set_trace()
        exit(-1)

    hierRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    hierRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        hierRCNN.cuda()

    start = time.time()

    num_images = len(imdb.image_index)

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                             imdb.num_classes, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)

    data_iter = iter(dataloader)

    hierRCNN.eval()

    use_rpn = False
    TP_count = 0.0
    TP_score = 0.0
    N_count = 0.1

    det_roidb = {}
    gt_roidb = {}
    for i in range(num_images):
        print('test [%d/%d]' % (num_images, i+1))
        data = next(data_iter)
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])
        im_id = data[4][0]

        gt_roidb[im_id] = gt_boxes[0].cpu().data.numpy()
        gt_roidb[im_id][:, :4] = gt_roidb[im_id][:, :4] / data[1][0][2].item()

        det_tic = time.time()

        with torch.no_grad():
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = hierRCNN(im_data, im_info, gt_boxes, num_boxes, use_rpn)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if not use_rpn:
            raw_label_inds = objnet.get_raw_indexes()

            for ppp in range(scores.size()[1]):
                N_count += 1

                gt_cate = gt_boxes[0, ppp, 4].cpu().data.numpy()
                gt_node = objnet.get_node_by_index(int(gt_cate))
                all_scores = scores[0][ppp].cpu().data.numpy()

                # print('==== %s ====' % gt_node.name())
                #ranked_inds = np.argsort(all_scores)[::-1][:20]
                #sorted_scrs = np.sort(all_scores)[::-1][:20]
                # for item in zip(ranked_inds, sorted_scrs):
                #     print('%s (%.2f)' % (objnet.get_node_by_index(item[0]).name(), item[1]))

                tree = InferTree(objnet, all_scores)
                top_1 = tree.top_k(1)
                pred_cate = top_1[0][0]
                pred_scr = top_1[0][1]

                eval_scr = gt_node.score(pred_cate)
                pred_node = objnet.get_node_by_index(pred_cate)
                info = ('%s -> %s(%.2f)' % (gt_node.name(), pred_node.name(),eval_scr))
                if eval_scr > 0:
                    TP_score += eval_scr
                    TP_count += 1
                    info = 'T: ' + info
                else:
                    info = 'F: ' + info
                    pass
                print(info)

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                    print(imdb.classes)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    print(imdb.classes)
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2].item()

        pred_scores = scores[0]
        pred_boxes = pred_boxes[0]

        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()

        # x1,y1,x2,y2,s0,s1,s2,s3,...,sN
        my_dets = torch.cat([pred_boxes[:, :4], pred_scores], 1)
        det_roidb[im_id] = my_dets.cpu().data.numpy()

    with open('det_roidb_%s.bin' % args.dataset, 'wb') as f:
        pickle.dump(det_roidb, f)
    #with open('gt_roidb_%s.bin' % args.dataset, 'wb') as f:
    #    pickle.dump(gt_roidb, f)

    end = time.time()
    print("test time: %0.4fs" % (end - start))

    print("Rec flat Acc: %.4f" % (TP_count / N_count))
    print("Rec hier Acc: %.4f" % (TP_score / N_count))
