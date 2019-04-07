import cv2
import os
import json
from matplotlib import pyplot as plt



def show_boxes(im, dets, cls, confs, mode='single'):
    """Draw detected bounding boxes."""
    for i in range(0, len(dets)):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        bbox = dets[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2],
                          bbox[3], fill=False,
                          edgecolor='red', linewidth=1.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '%s (%.2f)' % (cls[i], confs[i]),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
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
