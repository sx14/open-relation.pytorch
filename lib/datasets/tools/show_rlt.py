import cv2
import os
import json
from matplotlib import pyplot as plt
from global_config import VG_ROOT, VRD_ROOT


def show_rlts(im, rlt_boxes, rlt_cls):
    """Draw relationship"""
    for i in range(len(rlt_boxes[0])):
        # for each relationship
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')

        for p in range(len(rlt_boxes)):
            # for each component
            bbox = rlt_boxes[p][i]
            cls = rlt_cls[p][i]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2],
                              bbox[3], fill=False,
                              edgecolor='red', linewidth=1.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    '{}'.format(cls),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def get_rlts(img_root, anno_root, img_name):
    img_path = os.path.join(img_root, img_name)
    im = cv2.imread(img_path)
    anno_path = os.path.join(anno_root, img_name.split('.')[0] + '.json')
    with open(anno_path, 'r') as anno_file:
        anno = json.load(anno_file)
    rlts = anno['relationships']

    pso_cls = [[], [], []]  # pre_cls, sbj_cls, obj_cls
    pso_boxes = [[], [], []]

    for rlt in rlts:
        pre = rlt['predicate']
        sbj = rlt['subject']
        obj = rlt['object']

        parts = [pre, sbj, obj]
        for i in range(len(parts)):
            part = parts[i]
            pso_cls[i].append(part['name'])
            pso_boxes[i].append([part['xmin'], part['ymin'], part['xmax']-part['xmin'], part['ymax']-part['ymin']])

    return im, pso_cls, pso_boxes


if __name__ == '__main__':
    dataset = 'vg'

    if dataset == 'vg':
        dataset_root = VG_ROOT
    else:
        dataset_root = VRD_ROOT

    img_root = os.path.join(dataset_root, 'JPEGImages')
    anno_root = os.path.join(dataset_root, 'anno')

    for img_name in os.listdir(img_root):

        img_id = img_name.split('.')[0]
        anno_path = os.path.join(anno_root, img_id+'.json')
        if not os.path.exists(anno_path):
            continue

        im, cls, boxes = get_rlts(img_root, anno_root, img_name)

        show_rlts(im, boxes, cls)
