# coding=utf-8
import numpy as np
import cv2
import torch
from torch.autograd import Variable
from scipy.misc import imread

def get_raw_pred(all_scores, raw_inds):
    ranked_inds = np.argsort(all_scores)[::-1]
    pred_raw_ind = -1
    for ind in ranked_inds:
        if ind in raw_inds:
            pred_raw_ind = ind
            break
    assert pred_raw_ind > -1
    return pred_raw_ind, all_scores[pred_raw_ind]



def im_list_to_blob(ims):
    # 将resize后的图像转为输入网络的数据格式
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob


def get_roi_blob(boxes, scale):
    # ox1, oy1, ox2, oy2, cls, conf
    boxes_np = np.array(boxes)
    boxes_np[:, :4] = boxes_np[:, :4] * scale
    boxes_np = boxes_np[np.newaxis, :, :]
    return boxes_np

def get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
  TEST_SCALES = (600,)
  TEST_MAX_SIZE = 1000

  im_orig = im.astype(np.float32, copy=True)
  im_orig -= PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in TEST_SCALES:
    # TEST.SCALE 输入网络的图像短边长度
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    # 如果resize后图像的长边超过最大尺寸
    if np.round(im_scale * im_size_max) > TEST_MAX_SIZE:
      # 将图像按MAX_SIZE再次resize
      im_scale = float(TEST_MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)


def get_input_data(im, rois):
    im_blob, im_scales = get_image_blob(im)
    im_boxes = get_roi_blob(rois, im_scales[0])

    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)
    im_boxes_pt = torch.from_numpy(im_boxes)
    im_nbox_pt = torch.Tensor([im_boxes_pt.shape[1]])

    data=[im_data_pt, im_info_pt, im_boxes_pt, im_nbox_pt, im_scales[0]]
    return data
