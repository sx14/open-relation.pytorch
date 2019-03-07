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
        from lib.datasets.vg.label_hier.obj_hier import objnet
        from lib.datasets.vg.label_hier.pre_hier import prenet
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
    hierRCNN = vgg16_det(objnet, obj_vec_path)
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
    hierRela = HierRela(hierVis, None, objconf.label_vec_path())
    if args.cuda:
        hierRela.cuda()
    hierRela.eval()
    print('load model successfully!')

    # Initilize the tensor holder here.
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

    # Load gt data
    gt_roidb_path = os.path.join(PROJECT_ROOT, 'hier_rela', 'gt_box_label_%s.bin' % args.dataset)
    with open(gt_roidb_path, 'rb') as f:
        gt_roidb = pickle.load(f)

    N_count = 1.0
    TP_score = 0.0
    TP_count = 0.0

    pred_roidb = {}
    start = time.time()
    for img_id in gt_roidb:

        img_path = os.path.join(img_root, '%s.jpg' % img_id)
        img = cv2.imread(img_path)
        gt_rois = gt_roidb[img_id]

        data = get_input_data(img, gt_rois)

        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_relas.data.resize_(data[2].size()).copy_(data[2])
        num_relas.data.resize_(data[3].size()).copy_(data[3])

        im_scale = data[4]

        det_tic = time.time()
        rois, cls_score, \
        _, rois_label = hierRela(im_data, im_info, gt_relas, num_relas)

        scores = cls_score.data

        pred_cates = torch.zeros(rois[0].shape[0])
        pred_scores = torch.zeros(rois[0].shape[0], 1)

        raw_label_inds = prenet.get_raw_indexes()
        for ppp in range(scores.size()[1]):
            N_count += 1

            gt_cate = gt_relas[0, ppp, 4].cpu().data.numpy()
            gt_node = prenet.get_node_by_index(int(gt_cate))
            all_scores = scores[0][ppp].cpu().data.numpy()

            # print('==== %s ====' % gt_node.name())
            # ranked_inds = np.argsort(all_scores)[::-1][:20]
            # sorted_scrs = np.sort(all_scores)[::-1][:20]
            # for item in zip(ranked_inds, sorted_scrs):
            #     print('%s (%.2f)' % (objnet.get_node_by_index(item[0]).name(), item[1]))

            top2 = my_infer(prenet, all_scores)
            pred_cate = top2[0][0]
            pred_scr = top2[0][1]


            pred_cates[ppp] = pred_cate
            pred_scores[ppp] = pred_scr

            eval_scr = gt_node.score(pred_cate)
            pred_node = prenet.get_node_by_index(pred_cate)
            info = ('%s -> %s(%.2f)' % (gt_node.name(), pred_node.name(), eval_scr))
            if eval_scr > 0:
                TP_count += 1
                TP_score += eval_scr
                info = 'T: ' + info
            else:
                info = 'F: ' + info
                pass
            print(info)

        pred = torch.from_numpy(gt_rois)
        pred[:, 4] = pred_cate
        pred = torch.cat((pred, pred_scores), dim=1)
        pred_roidb[img_id] = pred.numpy()



    end = time.time()
    print("test time: %0.4fs" % (end - start))

    print("Rec flat Acc: %.4f" % (TP_count / N_count))
    print("Rec heir Acc: %.4f" % (TP_score / N_count))

    pred_roidb_path = os.path.join(PROJECT_ROOT, 'hier_rela', 'pre_box_label_%s.bin' % args.dataset)
    with open(pred_roidb_path, 'wb') as f:
        pickle.dump(pred_roidb, f)

