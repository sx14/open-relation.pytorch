from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pprint

from torch.autograd import Variable

from global_config import HierLabelConfig
from hier_det import infer
from lib.model.hier_rcnn.vgg16 import vgg16 as vgg16_det
from lib.model.hier_rela.hier_rela import HierRela
from lib.model.hier_rela.lang.hier_lang import HierLang
from lib.model.hier_rela.visual.vgg16 import vgg16 as vgg16_rela
from lib.model.hier_utils.helpers import *
from lib.model.hier_utils.infer_tree import InferTree
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

args = None
hierRela = None


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


def load_model():
    global args, hierRela
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.dataset == "vg":
        args.imdb_name = "vg_2016_trainval"
        args.imdbval_name = "vg_2016_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        from lib.datasets.vg200.label_hier.obj_hier import objnet
        from lib.datasets.vg200.label_hier.pre_hier import prenet

    elif args.dataset == "vrd":
        args.imdb_name = "vrd_2016_trainval"
        args.imdbval_name = "vrd_2016_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        from lib.datasets.vrd.label_hier.obj_hier import objnet
        from lib.datasets.vrd.label_hier.pre_hier import prenet

    if args.cuda:
        cfg.CUDA = True

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


def infer(batch, scale=True):
    with torch.no_grad():
        if args.dataset == "vg":
            from lib.datasets.vg200.label_hier.obj_hier import objnet
            from lib.datasets.vg200.label_hier.pre_hier import prenet

        elif args.dataset == "vrd":
            from lib.datasets.vrd.label_hier.obj_hier import objnet
            from lib.datasets.vrd.label_hier.pre_hier import prenet

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
        det_roidb = infer.infer(batch, scale=False)
        cond_roidb = gen_rela_conds(det_roidb)
        rela_roidb_use = cond_roidb

        N_count = 1e-10
        zero_N_count = 1e-10

        pred_roidb = {}
        N_img = len(rela_roidb_use.keys())
        for i, img_id in enumerate(rela_roidb_use.keys()):
            print('pred [%d/%d] %s' % (N_img, i + 1, img_id))
            # use unique object pairs
            img = batch[batch.keys()[i]]
            rois_use = rela_roidb_use[img_id]
            rois_use = np.array(rois_use)
            rois_use[:, 4] = 0
            rois_use_uni = np.unique(rois_use, axis=0)
            rois_use = rois_use_uni.tolist()

            # Attention: resized image data
            data = get_input_data(img, rois_use)

            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            relas_box.data.resize_(data[2].size()).copy_(data[2])
            relas_num.data.resize_(data[3].size()).copy_(data[3])
            relas_zero = np.array(rois_use)[:, -1]

            with torch.no_grad():
                rois, cls_score, \
                _, rois_label, vis_score, lan_score = hierRela(im_data, im_info, relas_box, relas_num)

            scores = cls_score.data

            pred_cates = torch.zeros(rois[0].shape[0], 4)
            pred_scores = torch.zeros(rois[0].shape[0], 4)

            hit = torch.zeros(scores.size()[1], 1)
            for ppp in range(scores.size()[1]):
                N_count += 1

                if relas_zero[ppp] == 1:
                    zero_N_count += 1

                all_scores = scores[0][ppp].cpu().data.numpy()

                top4 = InferTree(prenet, all_scores).top_k(4)
                for t in range(4):
                    pred_cate = top4[t][0]
                    pred_scr = top4[t][1]

                    # raw_cate, raw_score = get_raw_pred(all_scores, raw_label_inds, t + 1)

                    pred_cates[ppp, t] = pred_cate
                    pred_scores[ppp, t] = float(pred_scr)

            img_pred_rois = None
            for t in range(4):
                pred_rois = torch.FloatTensor(rois_use)
                sbj_scores = pred_rois[:, 16]
                obj_scores = pred_rois[:, 17]

                rela_scores = pred_scores[:, t] * sbj_scores * obj_scores
                rela_scores = rela_scores
                rela_indexes = np.argsort(rela_scores.numpy())[::-1]
                rela_scores = rela_scores.unsqueeze(1)

                pred_rois[:, 4] = pred_cates[:, t]
                # remove [pconf, sconf, oconf], cat rela_conf, hit
                pred_rois = torch.cat((pred_rois[:, :15], rela_scores, hit), dim=1)
                pred_rois = pred_rois.numpy()
                pred_rois = pred_rois[rela_indexes, :]

                if img_pred_rois is None:
                    img_pred_rois = pred_rois
                else:
                    img_pred_rois = np.concatenate((img_pred_rois, pred_rois), axis=0)
            if scale:
                img_pred_rois[:, [0, 2, 5, 7, 10, 12]] = img_pred_rois[:, [0, 2, 5, 7, 10, 12]] / img.shape[1]
                img_pred_rois[:, [1, 3, 6, 8, 11, 13]] = img_pred_rois[:, [1, 3, 6, 8, 11, 13]] / img.shape[0]
            pred_roidb[img_id] = img_pred_rois.tolist()
            # px1, py1, px2, py2, pcls, sx1, sy1, sx2, sy2, scls, ox1, oy1, ox2, oy2, ocls, rela_conf, hit
        return pred_roidb
