import numpy as np
import cv2
import caffe
from lib.fast_rcnn.test import im_detect
from open_relation import global_config


# load cnn
prototxt = global_config.fast_prototxt_path
caffemodel = global_config.fast_caffemodel_path
datasets = ['train', 'test']
caffe.set_mode_gpu()
caffe.set_device(0)
cnn = caffe.Net(prototxt, caffemodel, caffe.TEST)


def ext_cnn_feat(im, boxes):
    im_detect(cnn, im, boxes)
    fc7s = np.array(cnn.blobs['fc7'].data)
    return fc7s