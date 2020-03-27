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
from lib.roi_data_layer.hierRoibatchLoader import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.nms.nms_wrapper import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv
from lib.model.utils.net_utils import vis_detections
from lib.model.hier_rela.visual.vgg16 import vgg16 as vgg16_rela
from lib.model.hier_rcnn.vgg16 import vgg16 as vgg16_det
from global_config import HierLabelConfig

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
                        default='vg', type=str)
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
                        help='directory to load models', default="output",
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
                        default=100, type=int)
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
        args.imdb_name = "vg_lsj_2007_trainval"
        args.imdbval_name = "vg_lsj_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        from lib.datasets.vglsj.label_hier.obj_hier import objnet
        from lib.datasets.vglsj.label_hier.pre_hier import prenet

        args.class_agnostic = True

    elif args.dataset == "vrd":
        args.imdb_name = "vrd_2016_trainval"
        args.imdbval_name = "vrd_2016_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        from lib.datasets.vrd.label_hier.obj_hier import objnet
        from lib.datasets.vrd.label_hier.pre_hier import prenet

        args.class_agnostic = True

    args.cfg_file = "../../cfgs/{}_ls.yml".format(args.net) if args.large_scale else "../../cfgs/{}.yml".format(
        args.net)

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
                             'hier_rela_spa_{}_{}_{}_{}.pth'.
                             format(args.checksession, args.checkepoch, args.checkpoint, args.dataset))

    # initilize the network here.
    objconf = HierLabelConfig(args.dataset, 'object')
    obj_vec_path = objconf.label_vec_path()
    hierRCNN = vgg16_det(objnet, obj_vec_path, class_agnostic=True)
    hierRCNN.create_architecture()

    preconf = HierLabelConfig(args.dataset, 'predicate')
    pre_vec_path = preconf.label_vec_path()
    hierVis = vgg16_rela(prenet, pre_vec_path, hierRCNN)
    hierVis.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    hierVis.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    gt_relas = torch.FloatTensor(1)
    num_relas = torch.LongTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_relas = num_relas.cuda()
        gt_relas = gt_relas.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_relas = Variable(num_relas)
    gt_relas = Variable(gt_relas)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        hierRCNN.cuda()
        hierVis.cuda()

    start = time.time()

    max_per_image = 100

    vis = args.vis

    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    save_name = 'hier_rela_rcnn_10'
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                             imdb.num_classes, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    hierVis.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    use_rpn = False
    TP_count = 0.0
    N_count = 0.1

    for i in range(num_images):

        data = next(data_iter)
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_relas.data.resize_(data[2].size()).copy_(data[2])
        num_relas.data.resize_(data[3].size()).copy_(data[3])

        det_tic = time.time()
        rois, cls_score, \
        _, rois_label = hierVis(im_data, im_info, gt_relas, num_relas)

        scores = cls_score.data
        boxes = rois.data[:, :, 1:5]

        if not use_rpn:
            raw_label_inds = prenet.get_raw_indexes()
            for ppp in range(scores.size()[1]):
                N_count += 1
                gt_cate = gt_relas[0, ppp, 4].cpu().data.numpy()
                gt_node = prenet.get_node_by_index(int(gt_cate))
                all_scores = scores[0][ppp].cpu().data.numpy()
                ranked_inds = np.argsort(all_scores)[::-1][:20]
                sorted_scrs = np.sort(all_scores)[::-1][:20]
                raw_scores = all_scores[raw_label_inds]
                pred_raw_ind = np.argmax(raw_scores[1:]) + 1
                pred_cate = raw_label_inds[pred_raw_ind]
                gt_cate = gt_relas[0, ppp, 4].cpu().data.numpy()
                pred_node = prenet.get_node_by_index(pred_cate)
                gt_node = prenet.get_node_by_index(int(gt_cate))
                info = ('%s -> %s(%.2f)' % (gt_node.name(), pred_node.name(), raw_scores[pred_raw_ind]))
                if pred_cate == gt_cate:
                    # flat recall
                    TP_count += 1
                    info = 'T: ' + info
                else:
                    info = 'F: ' + info
                    pass
                print(info)

        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2].item()

        scores = scores[0]
        pred_boxes = pred_boxes[0]

        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)

        for j in xrange(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                all_boxes[j][i] = cls_dets.cpu().numpy()

            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

    end = time.time()
    print("test time: %0.4fs" % (end - start))

    print("Rec flat Acc: %.4f" % (TP_count / N_count))
