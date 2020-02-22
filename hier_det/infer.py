from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pdb
import pprint

from torch.autograd import Variable

from global_config import HierLabelConfig
from lib.model.faster_rcnn.resnet import resnet
from lib.model.faster_rcnn.vgg16 import vgg16
from lib.model.hier_rcnn.resnet import resnet as res101_det
from lib.model.hier_rcnn.vgg16 import vgg16 as vgg16_det
from lib.model.hier_utils.helpers import *
from lib.model.hier_utils.infer_tree import InferTree
from lib.model.nms.nms_wrapper import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv
from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.utils.blob import im_list_to_blob
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

fasterRCNN = None
hierRCNN = None
fatser_args = None
hier_args = None


def parse_faster_args():
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
                        help='vgg16, res101',
                        default='vgg16', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="../hier_det/fast_output",
                        type=str)
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true',
                        default=True)
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
                        default=73793, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')

    args = parser.parse_args()
    return args


def parse_hier_args():
    parser = argparse.ArgumentParser(description='Train a Hier R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='vrd', type=str)
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
                        default=20, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=7547, type=int)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="../hier_det/hier_output",
                        type=str)
    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def load_faster_model():
    global fasterRCNN
    global faster_args
    args = parse_faster_args()
    faster_args = args

    print('Called with args:')
    print(args)
    if args.dataset == "vg":
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        from lib.datasets.vg200.label_hier.obj_hier import objnet
    elif args.dataset == "vrd":
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        from lib.datasets.vrd.label_hier.obj_hier import objnet

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(objnet.get_raw_labels(), pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(objnet.get_raw_labels(), 101, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # pdb.set_trace()

    print("load checkpoint %s" % (load_name))


def load_hier_model():
    global hier_args
    global hierRCNN
    args = parse_hier_args()
    hier_args = args

    print('Called with args:')
    print(hier_args)

    if args.dataset == "vg":
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        from lib.datasets.vg200.label_hier.obj_hier import objnet
    elif args.dataset == "vrd":
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        from lib.datasets.vrd.label_hier.obj_hier import objnet

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

    if args.net == 'vgg16':
        hierRCNN = vgg16_det(objnet, objconf.label_vec_path(), class_agnostic=True)
        hierRCNN.create_architecture()
    elif args.net == 'res101':
        hierRCNN = res101_det(objnet, objconf.label_vec_path(), class_agnostic=True)
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


def infer(batch, scale=True):
    with torch.no_grad():
        if faster_args.dataset == "vg":
            faster_args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                    '50']
            from lib.datasets.vg200.label_hier.obj_hier import objnet


        elif faster_args.dataset == "vrd":
            faster_args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
            from lib.datasets.vrd.label_hier.obj_hier import objnet

        # initilize the tensor holder here.
        im_data = torch.FloatTensor(1)
        im_info = torch.FloatTensor(1)
        num_boxes = torch.LongTensor(1)
        gt_boxes = torch.FloatTensor(1)

        # ship to cuda
        if faster_args.cuda:
            im_data = im_data.cuda()
            im_info = im_info.cuda()
            num_boxes = num_boxes.cuda()
            gt_boxes = gt_boxes.cuda()

        # make variable
        with torch.no_grad():
            im_data = Variable(im_data)
            im_info = Variable(im_info)
            num_boxes = Variable(num_boxes)
            gt_boxes = Variable(gt_boxes)

        if faster_args.cuda:
            cfg.CUDA = True

        if faster_args.cuda:
            fasterRCNN.cuda()

        fasterRCNN.eval()

        thresh = 0.4

        num_images = len(batch.keys())

        print('Loaded Photo: {} images.'.format(num_images))

        det_roidb = {}
        while (num_images > 0):
            num_images -= 1
            im_in = batch[batch.keys()[num_images]]
            if len(im_in.shape) == 2:
                im_in = im_in[:, :, np.newaxis]
                im_in = np.concatenate((im_in, im_in, im_in), axis=2)
            # rgb -> bgr
            im_in = im_in[:, :, ::-1]
            im = im_in

            blobs, im_scales = _get_image_blob(im)
            assert len(im_scales) == 1, "Only single-image batch implemented"
            im_blob = blobs
            im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

            im_data_pt = torch.from_numpy(im_blob)
            im_data_pt = im_data_pt.permute(0, 3, 1, 2)
            im_info_pt = torch.from_numpy(im_info_np)

            im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.data.resize_(1, 1, 5).zero_()
            num_boxes.data.resize_(1).zero_()

            # pdb.set_trace()

            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    if faster_args.class_agnostic:
                        if faster_args.cuda:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        if faster_args.cuda:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                        box_deltas = box_deltas.view(1, -1, 4 * len(objnet.get_raw_labels()))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                _ = torch.from_numpy(np.tile(boxes, (1, scores.shape[1])))
                pred_boxes = _.cuda() if faster_args.cuda > 0 else _

            pred_boxes /= im_scales[0]

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()

            img_dets = []
            for j in xrange(1, len(objnet.get_raw_labels())):
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if faster_args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_cate = torch.FloatTensor(len(cls_scores))
                    cls_cate.fill_(j)
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    cls_dets_np_arr = cls_dets.cpu().data.numpy()
                    img_dets += cls_dets_np_arr.tolist()
            img_dets = np.array(img_dets)
            det_roidb[num_images] = img_dets

        if hier_args.dataset == "vg":
            faster_args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                    '50']
            from lib.datasets.vg200.label_hier.obj_hier import objnet


        elif hier_args.dataset == "vrd":
            faster_args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
            from lib.datasets.vrd.label_hier.obj_hier import objnet

        # init tensor holder here.
        im_data = torch.FloatTensor(1)
        im_info = torch.FloatTensor(1)
        num_boxes = torch.LongTensor(1)
        im_boxes = torch.FloatTensor(1)

        # shift to cuda
        if hier_args.cuda:
            im_data = im_data.cuda()
            im_info = im_info.cuda()
            num_boxes = num_boxes.cuda()
            im_boxes = im_boxes.cuda()

        # make variable
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        im_boxes = Variable(im_boxes)

        if hier_args.cuda:
            cfg.CUDA = True

        if hier_args.cuda:
            hierRCNN.cuda()
        hierRCNN.eval()
        # load proposals

        pred_roidb = {}
        N_img = len(det_roidb.keys())
        for i, img_id in enumerate(det_roidb.keys()):
            print('pred [%d/%d]' % (N_img, i + 1))
            img = batch[batch.keys()[i]]
            rois_use = det_roidb[img_id]

            if rois_use.shape[0] == 0:
                continue

            # Attention: resized image data
            data = get_input_data(img, rois_use)

            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            im_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])
            im_scale = data[4]

            with torch.no_grad():
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label = hierRCNN(im_data, im_info, im_boxes[:, :, :5], num_boxes, use_rpn=False)

            scores = cls_prob.data
            cate_scr = []
            for ppp in range(scores.size()[1]):
                all_scores = scores[0][ppp].cpu().data.numpy()
                top_1 = InferTree(objnet, all_scores).top_k(1)
                pred_cate = top_1[0][0]
                pred_scr = top_1[0][1]
                cate_scr += [[top_1[0][1]]]
                im_boxes[0][ppp][4] = pred_cate

                print(objnet.get_node_by_index(pred_cate).name(), pred_scr)

            im_boxes[:, :, :4] = im_boxes[:, :, :4] / im_scale
            boxes = torch.cat((im_boxes[0], torch.FloatTensor(cate_scr).cuda()), dim=1)
            boxes = boxes.data.cpu().numpy()
            if scale:
                boxes[:, 0] = boxes[:, 0] / img.shape[1]
                boxes[:, 1] = boxes[:, 1] / img.shape[0]
                boxes[:, 2] = boxes[:, 2] / img.shape[1]
                boxes[:, 3] = boxes[:, 3] / img.shape[0]
            pred_roidb[batch.keys()[i]] = boxes
        return pred_roidb
