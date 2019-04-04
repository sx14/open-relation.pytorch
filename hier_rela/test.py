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
from lib.model.hier_rcnn.vgg16 import vgg16 as vgg16_det
from lib.model.hier_rela.lang.hier_lang import HierLang
from lib.model.hier_rela.hier_rela import HierRela
from lib.model.hier_utils.tree_infer1 import my_infer
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
    parser.add_argument('--mode', dest='mode',
                        help='Do predicate recognition or relationship detection?',
                        action='store_true',
                        default='rela')


    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.dataset == "vg":
        args.imdb_name = "vg_2016_trainval"
        args.imdbval_name = "vg_2016_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        from lib.datasets.vg200.label_hier.obj_hier import objnet
        from lib.datasets.vg200.label_hier.pre_hier import prenet
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
    hierRCNN = vgg16_det(objnet, objconf.label_vec_path(), class_agnostic=True)
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
    hierLan = HierLang(600 * 2, preconf.label_vec_path())
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

    # shift to cuda
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

    if args.mode == 'pre':
        # Load gt data
        gt_roidb_path = os.path.join(PROJECT_ROOT, 'hier_rela', 'gt_rela_roidb_%s.bin' % args.dataset)
        with open(gt_roidb_path, 'rb') as f:
            gt_roidb = pickle.load(f)
            rela_roidb_use = gt_roidb
    else:
        det_roidb_path = os.path.join(PROJECT_ROOT, 'hier_rela', 'det_roidb_%s.bin' % args.dataset)
        with open(det_roidb_path, 'rb') as f:
            det_roidb = pickle.load(f)
        cond_roidb = gen_rela_conds(det_roidb)
        rela_roidb_use = cond_roidb

    N_count = 1e-10
    flat_count = 0.0
    hier_score_sum = 0.0
    raw_score_sum = 0.0
    infer_score_sum = 0.0

    raw_score_sum_u = 0.0

    zero_N_count = 1e-10
    zero_flat_count = 0.0
    zero_hier_score_sum = 0.0
    zero_raw_score_sum = 0.0
    zero_infer_score_sum = 0.0



    pred_roidb = {}
    start = time.time()
    N_img = len(rela_roidb_use.keys())
    for i, img_id in enumerate(rela_roidb_use.keys()):
        print('pred [%d/%d] %s' % (N_img, i + 1, img_id))
        img_path = os.path.join(img_root, '%s.jpg' % img_id)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        rois_use = rela_roidb_use[img_id]

        # Attention: resized image data
        data = get_input_data(img, rois_use)

        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        relas_box.data.resize_(data[2].size()).copy_(data[2])
        relas_num.data.resize_(data[3].size()).copy_(data[3])
        relas_zero = np.array(rois_use)[:, -1]

        im_scale = data[4]

        det_tic = time.time()
        with torch.no_grad():
            rois, cls_score, \
            _, rois_label, vis_score, lan_score = hierRela(im_data, im_info, relas_box, relas_num)

        scores = cls_score.data
        vis_scores = vis_score.data
        lan_scores = lan_score.data

        pred_cates = torch.zeros(rois[0].shape[0])
        pred_scores = torch.zeros(rois[0].shape[0])

        raw_label_inds = set(prenet.get_raw_indexes())

        hit = torch.zeros(scores.size()[1], 1)
        for ppp in range(scores.size()[1]):
            N_count += 1

            if relas_zero[ppp] == 1:
                zero_N_count += 1

            all_scores = scores[0][ppp].cpu().data.numpy()
            l_scores = lan_scores[0][ppp].cpu().data.numpy()
            v_scores = vis_scores[0][ppp].cpu().data.numpy()

            # print('==== %s ====' % gt_node.name())
            # ranked_inds = np.argsort(all_scores)[::-1][:20]
            # sorted_scrs = np.sort(all_scores)[::-1][:20]
            # for item in zip(ranked_inds, sorted_scrs):
            #     print('%s (%.2f)' % (prenet.get_node_by_index(item[0]).name(), item[1]))

            gt_cate = relas_box[0, ppp, 4].cpu().data.numpy()
            gt_node = prenet.get_node_by_index(int(gt_cate))

            top2 = my_infer(prenet, all_scores)
            pred_cate = top2[0][0]
            pred_scr = top2[0][1]

            raw_cate, raw_score = get_raw_pred(all_scores, raw_label_inds)
            vis_cate, _ = get_raw_pred(v_scores, raw_label_inds)
            lan_cate, _ = get_raw_pred(l_scores, raw_label_inds)

            pred_cates[ppp] = pred_cate
            pred_scores[ppp] = float(pred_scr)
            pred_node = prenet.get_node_by_index(pred_cate)

            if args.mode == 'pre':
                # print('=== %s ===' % gt_node.name())
                if vis_cate == gt_cate or lan_cate == gt_cate:
                    raw_score_sum_u += 1

                raw_node = prenet.get_node_by_index(raw_cate)
                vis_node = prenet.get_node_by_index(vis_cate)
                lan_node = prenet.get_node_by_index(lan_cate)

                if raw_cate == gt_cate:
                    raw_score_sum += 1

                inf_scr = gt_node.score(pred_cate)
                infer_score_sum += inf_scr

                hier_scr = gt_node.score(raw_cate)
                hier_score_sum += hier_scr

                if relas_zero[ppp] == 1:
                    if raw_cate == gt_cate:
                        zero_raw_score_sum += 1
                    zero_hier_score_sum += hier_scr
                    zero_infer_score_sum += inf_scr

                info = ('%s -> %s(%.2f)' % (gt_node.name(), pred_node.name(), inf_scr))
                # info = ('%s -> %s | %s' % (gt_node.name(), vis_node.name(), lan_node.name()))
                if inf_scr > 0:
                    hit[ppp, 0] = inf_scr
                    flat_count += 1
                    if relas_zero[ppp] == 1:
                        zero_flat_count += 1
                    info = 'T: ' + info
                else:
                    info = 'F: ' + info
                    pass
                print(info)

        pred_rois = torch.FloatTensor(rois_use)
        sbj_scores = pred_rois[:, -3]
        obj_scores = pred_rois[:, -2]
        rela_scores = pred_scores * sbj_scores * obj_scores
        rela_scores = rela_scores.unsqueeze(1)

        pred_rois[:, 4] = pred_cates
        # remove [pconf, sconf, oconf], cat rela_conf, hit
        pred_rois = torch.cat((pred_rois[:, :15], rela_scores, hit), dim=1)
        pred_roidb[img_id] = pred_rois.numpy()
        # px1, py1, px2, py2, pcls, sx1, sy1, sx2, sy2, scls, ox1, oy1, ox2, oy2, ocls, rela_conf, hit



    print("==== overall test result ==== ")
    print("Rec raw  Acc: %.4f" % (raw_score_sum / N_count))
    print("Rec raw_v+l  Acc: %.4f" % (raw_score_sum_u / N_count))
    print("Rec heir Acc: %.4f" % (hier_score_sum / N_count))
    print("Rec infer Acc: %.4f" % (infer_score_sum / N_count))
    print("Rec flat Acc: %.4f" % (flat_count / N_count))

    print("==== zero-shot test result ==== ")
    print("Rec raw  Acc: %.4f" % (zero_raw_score_sum / zero_N_count))
    print("Rec heir Acc: %.4f" % (zero_hier_score_sum / zero_N_count))
    print("Rec infer Acc: %.4f" % (zero_infer_score_sum / N_count))
    print("Rec flat Acc: %.4f" % (zero_flat_count / zero_N_count))

    pred_roidb_path = os.path.join(PROJECT_ROOT, 'hier_rela', '%s_box_label_%s_ours.bin' % (args.mode, args.dataset))
    with open(pred_roidb_path, 'wb') as f:
        pickle.dump(pred_roidb, f)

