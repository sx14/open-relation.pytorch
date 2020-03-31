from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle
import pprint
import time

from torch.autograd import Variable

from global_config import HierLabelConfig
from global_config import PROJECT_ROOT, VG_ROOT, VRD_ROOT
from lib.model.hier_rcnn.vgg16 import vgg16 as vgg16_det
from lib.model.hier_utils.helpers import *
from lib.model.hier_utils.infer_tree import InferTree
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list

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
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='vgg16', type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true',
                        default=True)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=6, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=12959, type=int)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="hier_output_new",
                        type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.dataset == "vg":
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        from lib.datasets.vglsj.label_hier.obj_hier import objnet

        img_root = os.path.join(VG_ROOT, 'JPEGImages')

    elif args.dataset == "vrd":
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        from lib.datasets.vrd.label_hier.obj_hier import objnet

        img_root = os.path.join(VRD_ROOT, 'JPEGImages')

    args.cfg_file = "../cfgs/vgg16.yml"

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    # init the network here.
    # load detector
    objconf = HierLabelConfig(args.dataset, 'object')
    obj_vec_path = objconf.label_vec_path()
    hierRCNN = vgg16_det(objnet, objconf.label_vec_path(), class_agnostic=True)
    hierRCNN.create_architecture()

    # load weights
    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'hier_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    hierRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')

    # init tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    im_boxes = torch.FloatTensor(1)

    # shift to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        im_boxes = im_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    im_boxes = Variable(im_boxes)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        hierRCNN.cuda()
    hierRCNN.eval()
    # load proposals
    det_roidb_path = os.path.join(PROJECT_ROOT, 'hier_det', 'det_roidb_hier_%s.bin' % args.dataset)
    with open(det_roidb_path, 'rb') as f:
        det_roidb = pickle.load(f)

    pred_roidb = {}
    start = time.time()
    N_img = len(det_roidb.keys())
    for i, img_id in enumerate(det_roidb.keys()):
        print('pred [%d/%d]' % (N_img, i + 1))
        img_path = os.path.join(img_root, '%s.jpg' % img_id)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        rois_use = det_roidb[img_id]

        if rois_use.shape[0] == 0:
            continue

        # Attention: resized image data
        data = get_input_data(img, rois_use, mode = 'det')

        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        im_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])
        im_scale = data[4]

        det_tic = time.time()
        with torch.no_grad():
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = hierRCNN(im_data, im_info, im_boxes[:, :, :5], num_boxes, use_rpn=False)

        scores = cls_prob.data

        for ppp in range(scores.size()[1]):
            all_scores = scores[0][ppp].cpu().data.numpy()
            top1 = InferTree(objnet, all_scores).top_k(1)
            pred_cate = top1[0][0]
            pred_scr = top1[0][1]
            im_boxes[0][ppp][4] = pred_cate
            im_boxes[0][ppp][5] = pred_scr

        im_boxes[:, :, :4] = im_boxes[:, :, :4] / im_scale
        pred_roidb[img_id] = im_boxes[0].data.cpu().numpy()

    pred_roidb_path = os.path.join(PROJECT_ROOT, 'hier_rela', 'det_roidb_hier_%s.bin' % args.dataset)
    with open(pred_roidb_path, 'wb') as f:
        pickle.dump(pred_roidb, f)
