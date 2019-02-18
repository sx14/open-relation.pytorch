# coding=utf-8
import numpy as np
import cv2
import torch
from torch.autograd import Variable
from scipy.misc import imread


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


def _get_image_blob(im):
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


def ext_cnn_feat(net, img_path, boxes):
    net.eval()
    # initilize the Tensor holder here.
    im_data = torch.FloatTensor(1).cuda()
    im_info = torch.FloatTensor(1).cuda()
    num_boxes = torch.LongTensor(1).cuda()
    gt_boxes = torch.FloatTensor(1).cuda()

    with torch.no_grad():
        # make variable
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)

    # process inputs
    im_in = np.array(imread(img_path))
    # rgb -> bgr
    im = im_in[:, :, ::-1]


    # 准备输入网络的图像数据
    blobs, im_scales = _get_image_blob(im)

    # 网络输入
    im_blob = blobs
    # 图像信息，图像高、宽、resize比例
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
    im_boxes_np = np.array(boxes)[np.newaxis, :, :]


    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)
    im_boxes_pt = torch.from_numpy(im_boxes_np)

    # 张量装进预先定义好的Variable
    im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
    im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
    gt_boxes.data.resize_(im_boxes_pt.size()).copy_(im_boxes_pt)
    num_boxes.data.resize_(1).zero_()
    num_boxes[0] = gt_boxes.size()[1]

    fc7 = net.ext_feat(im_data, im_info, gt_boxes, num_boxes)
    fc7 = fc7.cpu().data.numpy()
    return fc7
