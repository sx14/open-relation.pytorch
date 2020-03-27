# -*- coding: utf-8 -*-
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from lib.model.utils.config import cfg
from lib.roi_data_layer.hierMinibatch import get_minibatch
from lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import random
import time
import pdb


def bbox_trans(human_box_roi, object_box_roi, size=64):
    human_box = human_box_roi.copy()
    object_box = object_box_roi.copy()

    union_box = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                 max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]

    height = union_box[3] - union_box[1] + 1
    width = union_box[2] - union_box[0] + 1

    if height > width:
        ratio = 'height'
    else:
        ratio = 'width'

    # shift the top-left corner to (0,0)
    human_box[0] -= union_box[0]
    human_box[2] -= union_box[0]
    human_box[1] -= union_box[1]
    human_box[3] -= union_box[1]
    object_box[0] -= union_box[0]
    object_box[2] -= union_box[0]
    object_box[1] -= union_box[1]
    object_box[3] -= union_box[1]

    if ratio == 'height':  # height is larger than width

        human_box[0] = 0 + size * human_box[0] / height
        human_box[1] = 0 + size * human_box[1] / height
        human_box[2] = (size * width / height - 1) - size * (width - 1 - human_box[2]) / height
        human_box[3] = (size - 1) - size * (height - 1 - human_box[3]) / height

        object_box[0] = 0 + size * object_box[0] / height
        object_box[1] = 0 + size * object_box[1] / height
        object_box[2] = (size * width / height - 1) - size * (width - 1 - object_box[2]) / height
        object_box[3] = (size - 1) - size * (height - 1 - object_box[3]) / height

        # Need to shift horizontally
        union_box = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                     max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
        if human_box[3] > object_box[3]:
            human_box[3] = size - 1
        else:
            object_box[3] = size - 1

        shift = size / 2 - (union_box[2] + 1) / 2
        human_box += [shift, 0, shift, 0]
        object_box += [shift, 0, shift, 0]

    else:  # width is larger than height

        human_box[0] = 0 + size * human_box[0] / width
        human_box[1] = 0 + size * human_box[1] / width
        human_box[2] = (size - 1) - size * (width - 1 - human_box[2]) / width
        human_box[3] = (size * height / width - 1) - size * (height - 1 - human_box[3]) / width

        object_box[0] = 0 + size * object_box[0] / width
        object_box[1] = 0 + size * object_box[1] / width
        object_box[2] = (size - 1) - size * (width - 1 - object_box[2]) / width
        object_box[3] = (size * height / width - 1) - size * (height - 1 - object_box[3]) / width

        # Need to shift vertically
        union_box = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                     max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]

        if human_box[2] > object_box[2]:
            human_box[2] = size - 1
        else:
            object_box[2] = size - 1

        shift = size / 2 - (union_box[3] + 1) / 2

        human_box = human_box + [0, shift, 0, shift]
        object_box = object_box + [0, shift, 0, shift]

    return np.round(human_box), np.round(object_box)


def gen_spatial_map(human_box, object_box):
    hbox, obox = bbox_trans(human_box, object_box)
    spa_map = np.zeros((2, 64, 64), dtype='float32')
    spa_map[0, int(hbox[1]):int(hbox[3]) + 1, int(hbox[0]):int(hbox[2]) + 1] = 1
    spa_map[1, int(obox[1]):int(obox[3]) + 1, int(obox[0]):int(obox[2]) + 1] = 1
    return spa_map


class roibatchLoader(data.Dataset):
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
        self._roidb = roidb
        self._num_classes = num_classes
        # we make the height of image consistent to trim_height, trim_width
        self.trim_height = cfg.TRAIN.TRIM_HEIGHT
        self.trim_width = cfg.TRAIN.TRIM_WIDTH
        self.max_num_box = cfg.MAX_NUM_GT_BOXES
        self.training = training
        self.normalize = normalize
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.batch_size = batch_size
        self.data_size = len(self.ratio_list)

        # given the ratio_list, we want to make the ratio same for each batch.
        self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
        num_batch = int(np.ceil(len(ratio_index) / batch_size))
        for i in range(num_batch):
            left_idx = i * batch_size
            right_idx = min((i + 1) * batch_size - 1, self.data_size - 1)

            if ratio_list[right_idx] < 1:
                # for ratio < 1, we preserve the leftmost in each batch.
                target_ratio = ratio_list[left_idx]
            elif ratio_list[left_idx] > 1:
                # for ratio > 1, we preserve the rightmost in each batch.
                target_ratio = ratio_list[right_idx]
            else:
                # for ratio cross 1, we make it to be 1.
                target_ratio = 1

            self.ratio_list_batch[left_idx:(right_idx + 1)] = target_ratio

    def __getitem__(self, index):
        if self.training:
            index_ratio = int(self.ratio_index[index])
        else:
            index_ratio = index

        # get the anchor index for current sample index
        # here we set the anchor index to the last one
        # sample in this group

        # get one image
        minibatch_db = [self._roidb[index_ratio]]
        blobs = get_minibatch(minibatch_db, self._num_classes)
        data = torch.from_numpy(blobs['data'])
        im_info = torch.from_numpy(blobs['im_info'])

        num_box = blobs['gt_boxes'].shape[0]
        # we need to random shuffle the bounding box.
        data_height, data_width = data.size(1), data.size(2)
        if self.training:

            np.random.shuffle(blobs['gt_boxes'])
            # blob['gt_boxes'] = [px1, py1, py1, py2, pc, sx1, sx2, sy1, sy2, sc, ox1, ox2, oy1, oy2, oc]

            pre_boxes = blobs['gt_boxes'][:, 0:5]
            sbj_boxes = blobs['gt_boxes'][:, 5:10]
            obj_boxes = blobs['gt_boxes'][:, 10:15]

            # 沿dim=0连接pre,sbj,obj
            gt_boxes = np.concatenate((pre_boxes, sbj_boxes, obj_boxes), axis=0)
            gt_boxes = torch.from_numpy(gt_boxes)

            raw_spa_maps = np.zeros((num_box, 2, 64, 64))
            for i in range(num_box):
                raw_spa_maps[i] = gen_spatial_map(sbj_boxes[i, 0:4], obj_boxes[i][0:4])
            raw_spa_maps = np.tile(raw_spa_maps, (3, 1, 1, 1))
            gt_spa_maps = torch.from_numpy(raw_spa_maps).float()

            ########################################################
            # padding the input image to fixed size for each group #
            ########################################################

            # NOTE1: need to cope with the case where a group cover both conditions. (done)
            # NOTE2: need to consider the situation for the tail samples. (no worry)
            # NOTE3: need to implement a parallel data loader. (no worry)
            # get the index range

            # if the image need to crop, crop to the target size.
            ratio = self.ratio_list_batch[index]

            if self._roidb[index_ratio]['need_crop']:
                if ratio < 1:
                    # this means that data_width << data_height, we need to crop the
                    # data_height
                    # 图片太高了
                    # 把图像上方多余的部分剪掉
                    # 对应调整box
                    min_y = int(torch.min(gt_boxes[:, 1]))
                    max_y = int(torch.max(gt_boxes[:, 3]))

                    # resized img height
                    trim_size = int(np.floor(data_width / ratio))
                    if trim_size > data_height:
                        trim_size = data_height
                    box_region = max_y - min_y + 1
                    if min_y == 0:
                        # box已经顶到天了，无法裁剪上方部分
                        y_s = 0
                    else:
                        if (box_region - trim_size) < 0:
                            y_s_min = max(max_y - trim_size, 0)
                            y_s_max = min(min_y, data_height - trim_size)
                            if y_s_min == y_s_max:
                                y_s = y_s_min
                            else:
                                y_s = np.random.choice(range(y_s_min, y_s_max))
                        else:
                            y_s_add = int((box_region - trim_size) / 2)
                            if y_s_add == 0:
                                y_s = min_y
                            else:
                                y_s = np.random.choice(range(min_y, min_y + y_s_add))
                    # crop the image
                    data = data[:, y_s:(y_s + trim_size), :, :]

                    # shift y coordiante of gt_boxes
                    gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
                    gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)

                    # update gt bounding box according the trip
                    gt_boxes[:, 1].clamp_(0, trim_size - 1)
                    gt_boxes[:, 3].clamp_(0, trim_size - 1)

                else:
                    # this means that data_width >> data_height, we need to crop the
                    # data_width
                    # 图片太宽了
                    # 把图像左侧多余的部分剪掉
                    # 对应调整box
                    min_x = int(torch.min(gt_boxes[:, 0]))
                    max_x = int(torch.max(gt_boxes[:, 2]))

                    # resized img width
                    trim_size = int(np.ceil(data_height * ratio))
                    if trim_size > data_width:
                        trim_size = data_width
                    box_region = max_x - min_x + 1
                    if min_x == 0:
                        x_s = 0
                    else:
                        if (box_region - trim_size) < 0:
                            x_s_min = max(max_x - trim_size, 0)
                            x_s_max = min(min_x, data_width - trim_size)
                            if x_s_min == x_s_max:
                                x_s = x_s_min
                            else:
                                x_s = np.random.choice(range(x_s_min, x_s_max))
                        else:
                            x_s_add = int((box_region - trim_size) / 2)
                            if x_s_add == 0:
                                x_s = min_x
                            else:
                                x_s = np.random.choice(range(min_x, min_x + x_s_add))
                    # crop the image
                    data = data[:, :, x_s:(x_s + trim_size), :]

                    # shift x coordiante of gt_boxes
                    gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
                    gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
                    # update gt bounding box according the trip
                    gt_boxes[:, 0].clamp_(0, trim_size - 1)
                    gt_boxes[:, 2].clamp_(0, trim_size - 1)

            # based on the ratio, padding the image.
            if ratio < 1:
                # this means that data_width < data_height
                trim_size = int(np.floor(data_width / ratio))

                padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)), \
                                                 data_width, 3).zero_()

                padding_data[:data_height, :, :] = data[0]
                # update im_info
                im_info[0, 0] = padding_data.size(0)
                # print("height %d %d \n" %(index, anchor_idx))
            elif ratio > 1:
                # this means that data_width > data_height
                # if the image need to crop.
                padding_data = torch.FloatTensor(data_height, \
                                                 int(np.ceil(data_height * ratio)), 3).zero_()
                padding_data[:, :data_width, :] = data[0]
                im_info[0, 1] = padding_data.size(1)
            else:
                trim_size = min(data_height, data_width)
                padding_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
                padding_data = data[0][:trim_size, :trim_size, :]
                # gt_boxes.clamp_(0, trim_size)
                gt_boxes[:, :4].clamp_(0, trim_size)
                im_info[0, 0] = trim_size
                im_info[0, 1] = trim_size

            # check the bounding box:
            pre_boxes = gt_boxes[0:num_box, :]
            sbj_boxes = gt_boxes[num_box:num_box * 2, :]
            obj_boxes = gt_boxes[num_box * 2:num_box * 3, :]

            not_keep_pre = (pre_boxes[:, 0] == pre_boxes[:, 2]) | (pre_boxes[:, 1] == pre_boxes[:, 3])
            not_keep_sbj = (sbj_boxes[:, 0] == sbj_boxes[:, 2]) | (sbj_boxes[:, 1] == sbj_boxes[:, 3])
            not_keep_obj = (obj_boxes[:, 0] == obj_boxes[:, 2]) | (obj_boxes[:, 1] == obj_boxes[:, 3])
            # pre,sbj,obj 三者有任意一个box没有了就丢弃整个relationship
            not_keep = not_keep_pre | not_keep_sbj | not_keep_obj
            keep = torch.nonzero(not_keep == 0).view(-1)

            gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1) * 3).zero_()
            if keep.numel() != 0:
                pre_boxes = pre_boxes[keep]
                sbj_boxes = sbj_boxes[keep]
                obj_boxes = obj_boxes[keep]
                gt_spa_maps = gt_spa_maps[keep]

                gt_boxes = torch.cat([pre_boxes, sbj_boxes, obj_boxes], 1)

                num_boxes = min(gt_boxes.size(0), self.max_num_box)
                gt_boxes_padding[:num_boxes, :] = gt_boxes[:num_boxes]
                spa_maps_padding = gt_spa_maps[:num_boxes]
            else:
                spa_maps_padding = torch.LongTensor(1, 2, 64, 64).zero_()
                num_boxes = 0

                # permute trim_data to adapt to downstream processing
            padding_data = padding_data.permute(2, 0, 1).contiguous()
            im_info = im_info.view(3)

            return padding_data, im_info, gt_boxes_padding, spa_maps_padding, num_boxes
        else:
            data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
            im_info = im_info.view(3)
            gt_boxes = torch.from_numpy(blobs['gt_boxes'])
            sbj_boxes = blobs['gt_boxes'][:, 5:10]
            obj_boxes = blobs['gt_boxes'][:, 10:15]
            raw_spa_maps = np.zeros((num_box, 2, 64, 64))
            for i in range(num_box):
                raw_spa_maps[i] = gen_spatial_map(sbj_boxes[i, 0:4], obj_boxes[i][0:4])
            gt_spa_maps = torch.from_numpy(raw_spa_maps).float()
            num_boxes = gt_boxes.size(0)

            return data, im_info, gt_boxes, gt_spa_maps, num_boxes

    def __len__(self):
        return len(self._roidb)
