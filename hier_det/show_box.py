import cv2
import os
import json
from matplotlib import pyplot as plt
import random
import numpy as np

def show_boxes(im, dets, cls, confs, mode='single'):
    """Draw detected bounding boxes."""

    def random_color():
        color = []
        for i in range(3):
            color.append(random.randint(0, 255) / 255.0)
        return color

    if mode != 'single':
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')

    for i in range(0, len(dets)):
        if mode == 'single':
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(im, aspect='equal')

        bbox = dets[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2],
                          bbox[3], fill=False,
                          edgecolor=random_color(), linewidth=5)
        )
        # ax.text(bbox[0], bbox[1] - 2,
        #         '%s' % (cls[i]),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')
        if mode == 'single':
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    if mode != 'single':
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def get_pres(img_root, anno_root, img_name):
    img_path = os.path.join(img_root, img_name)
    im = cv2.imread(img_path)
    anno_path = os.path.join(anno_root, img_name.split('.')[0] + '.json')
    with open(anno_path, 'r') as anno_file:
        anno = json.load(anno_file)
    rlts = anno['relations']
    cls = []
    boxes = []
    for rlt in rlts:
        pre = rlt['predicate']
        cls.append(pre['name'])
        boxes.append([pre['xmin'], pre['ymin'], pre['xmax']-pre['xmin'], pre['ymax']-pre['ymin']])
    return im, cls, boxes

def get_objs(img_root, anno_root, img_name):
    img_path = os.path.join(img_root, img_name)
    im = cv2.imread(img_path)
    anno_path = os.path.join(anno_root, img_name.split('.')[0] + '.json')
    with open(anno_path, 'r') as anno_file:
        anno = json.load(anno_file)
    objs = anno['objects']
    cls = []
    boxes = []
    for obj in objs:
        cls.append(obj['name'])
        boxes.append([obj['xmin'], obj['ymin'], obj['xmax']-obj['xmin'], obj['ymax']-obj['ymin']])
    return im, cls, boxes


def draw_boxes(im, boxes, colors=None):

    def random_color():
        color = []
        for i in range(3):
            color.append(random.randint(0, 255))
        return color
    colors = get_colors()
    boxes = boxes.astype(np.int)
    default_color = np.array([0, 0, 255])
    if colors is None or len(colors) < len(boxes):
        colors_np = np.tile(default_color, (len(boxes), 1))
    else:
        colors_np = np.array(colors)
    im_dets = im.copy()
    for i, box in enumerate(boxes):
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h
        if i < len(colors):
            color = colors[i]
        else:
            color = random_color()
        im_dets = cv2.rectangle(im_dets, (x1, y1), (x2, y2), color, cv2.LINE_4, 0)
    return im_dets


def get_colors():
    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 255, 255]
    ]
    return np.array(colors)
