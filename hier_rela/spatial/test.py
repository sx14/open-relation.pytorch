from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import time

import numpy as np
import torch
from torch.autograd import Variable

from global_config import HierLabelConfig
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.roi_data_layer.hierRoibatchLoader import roibatchLoader
from lib.roi_data_layer.roidb import combined_roidb
from model.hier_rela.spatial.hier_spatial import HierSpatial
from lib.model.hier_utils.infer_tree import InferTree

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
                        default=20, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=12959, type=int)
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
        args.imdb_name = "vg_lsj_2016_trainval"
        args.imdbval_name = "vg_lsj_2016_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        from lib.datasets.vglsj.label_hier.pre_hier import prenet

        args.class_agnostic = True

    elif args.dataset == "vrd":
        args.imdb_name = "vrd_2016_trainval"
        args.imdbval_name = "vrd_2016_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
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
                             'hier_rela_spatial_{}_{}_{}_{}.pth'.
                             format(args.checksession, args.checkepoch, args.checkpoint, args.dataset))

    preconf = HierLabelConfig(args.dataset, 'predicate')
    pre_vec_path = preconf.label_vec_path()
    spaCNN = HierSpatial(prenet, pre_vec_path)

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    spaCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    gt_relas = torch.FloatTensor(1)
    spa_maps = torch.FloatTensor(1)
    num_relas = torch.LongTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_relas = num_relas.cuda()
        spa_maps = spa_maps.cuda()
        gt_relas = gt_relas.cuda()

    # make variable
    with torch.no_grad():
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_relas = Variable(num_relas)
        spa_maps = Variable(spa_maps)
        gt_relas = Variable(gt_relas)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        spaCNN.cuda()

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

    spaCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    use_rpn = False
    TP_count = 0.0
    N_count = 0.1
    TP_score = 0.0

    for i in range(num_images):

        data = next(data_iter)
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_relas.data.resize_(data[2].size()).copy_(data[2])
        spa_maps.data.resize_(data[3].size()).copy_(data[3])
        num_relas.data.resize_(data[4].size()).copy_(data[4])

        det_tic = time.time()
        pre_labels = gt_relas[0][:num_relas, 4]
        cls_score, RCNN_loss_cls = spaCNN(spa_maps[0], pre_labels)

        scores = cls_score.data

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

                top_1 = InferTree(prenet, all_scores).top_k(1)
                pred_cate = top_1[0][0]
                pred_scr = top_1[0][1]

                eval_scr = gt_node.score(pred_cate)
                pred_node = prenet.get_node_by_index(pred_cate)
                info = ('%s -> %s(%.2f)' % (
                    gt_node.name(), pred_node.name(), eval_scr))
                if eval_scr > 0:
                    TP_score += eval_scr
                    TP_count += 1
                    info = 'T: ' + info
                else:
                    info = 'F: ' + info
                    pass
                print(info)

    end = time.time()
    print("test time: %0.4fs" % (end - start))

    print("Rec flat Acc: %.4f" % (TP_count / N_count))
    print("Rec hier Acc: %.4f" % (TP_score / N_count))
