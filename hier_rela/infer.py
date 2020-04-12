from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pprint

from torch.autograd import Variable

from global_config import HierLabelConfig
from hier_det.infer import infer as det_infer
from lib.model.hier_rcnn.vgg16 import vgg16 as vgg16_det
from lib.model.hier_rela.hier_rela import HierRela
from lib.model.hier_rela.spatial.hier_spatial import HierSpatial
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
                        default='vg', type=str)
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
    parser.add_argument('--vis_checksession', dest='vis_checksession',
                        help='vis_checksession to load model',
                        default=1, type=int)
    parser.add_argument('--vis_checkepoch', dest='vis_checkepoch',
                        help='vis_checkepoch to load network',
                        default=10, type=int)
    parser.add_argument('--vis_checkpoint', dest='vis_checkpoint',
                        help='vis_checkpoint to load network',
                        default=12959, type=int)
    parser.add_argument('--spa_checksession', dest='spa_checksession',
                        help='spa_checksession to load model',
                        default=1, type=int)
    parser.add_argument('--spa_checkepoch', dest='spa_checkepoch',
                        help='spa_checkepoch to load network',
                        default=10, type=int)
    parser.add_argument('--spa_checkpoint', dest='spa_checkpoint',
                        help='spa_checkpoint to load network',
                        default=12959, type=int)
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
        from lib.datasets.vglsj.label_hier.obj_hier import objnet
        from lib.datasets.vglsj.label_hier.pre_hier import prenet

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
    load_name = '../hier_rela/visual/new_output/vgg16/{}/hier_rela_vis_{}_{}_{}_{}.pth'.format(args.dataset, args.vis_checksession,
                                                                                    args.vis_checkepoch,
                                                                                    args.vis_checkpoint, args.dataset)
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    hierVis.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    hierVis.eval()

    spaCNN = HierSpatial(prenet, pre_vec_path)
    load_name = '../hier_rela/spatial/output/vgg16/{}/hier_rela_spatial_{}_{}_{}_{}.pth'.format(args.dataset,
                                                                                     args.spa_checksession,
                                                                                     args.spa_checkepoch,
                                                                                     args.spa_checkpoint,
                                                                                     args.dataset)
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    spaCNN.load_state_dict(checkpoint['model'])
    spaCNN.eval()

    # get HierRela
    hierRela = HierRela(hierVis, spaCNN)
    if args.cuda:
        hierRela.cuda()
    hierRela.eval()
    print('load model successfully!')


def infer(batch, scale=True):
    with torch.no_grad():
        if args.dataset == "vg":
            from lib.datasets.vglsj.label_hier.obj_hier import objnet
            from lib.datasets.vglsj.label_hier.pre_hier import prenet

        elif args.dataset == "vrd":
            from lib.datasets.vrd.label_hier.obj_hier import objnet
            from lib.datasets.vrd.label_hier.pre_hier import prenet

        # Initilize the tensor holder here.
        im_data = torch.FloatTensor(1)
        im_info = torch.FloatTensor(1)
        relas_box = torch.FloatTensor(1)
        spa_maps = torch.FloatTensor(1)
        relas_num = torch.LongTensor(1)

        # shift to cuda
        if args.cuda:
            im_data = im_data.cuda()
            im_info = im_info.cuda()
            relas_num = relas_num.cuda()
            spa_maps = spa_maps.cuda()

            relas_box = relas_box.cuda()
        with torch.no_grad():
            # make variable
            im_data = Variable(im_data)
            im_info = Variable(im_info)
            relas_num = Variable(relas_num)
            relas_box = Variable(relas_box)
            spa_maps = Variable(spa_maps)

        det_roidb = det_infer(batch, scale=False)
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
            data = get_input_data(img, rois_use, mode='rela')

            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            relas_box.data.resize_(data[2].size()).copy_(data[2])
            spa_maps.data.resize_(data[3].size()).copy_(data[3])
            relas_num.data.resize_(data[4].size()).copy_(data[4])

            with torch.no_grad():
                rois, cls_score, \
                _, rois_label, vis_score, spa_score = hierRela(im_data, im_info, relas_box, spa_maps, relas_num)

            scores = cls_score.data

            pred_cates = torch.zeros(rois[0].shape[0], 4)
            pred_scores = torch.zeros(rois[0].shape[0], 4)

            hit = torch.zeros(scores.size()[1], 1)
            for ppp in range(scores.size()[1]):
                N_count += 1

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
