import os
import shutil
import json
import numpy as np
import cv2
from to_pascal_format import output_pascal_format
from path_config import vrd_root, vrd_pascal_root

def gen_JPEGImages(vrd_root, vrd_pascal_root):
    # output path
    JPEGImages_path = os.path.join(vrd_pascal_root, 'JPEGImages')
    if not os.path.isdir(JPEGImages_path):
        os.mkdir(JPEGImages_path)

    # original path
    vrd_img_dir_root = os.path.join(vrd_root, 'sg_dataset')
    for img_dir in os.listdir(vrd_img_dir_root):
        vrd_img_dir = os.path.join(vrd_img_dir_root, img_dir)
        for i, img_name in enumerate(os.listdir(vrd_img_dir)):
            if (i+1) % 1000 == 0:
                print(i+1)
            if not img_name.endswith('jpg'):
                # ATTENTION: only JPEG image is legal
                print(img_name + ' is discarded.')
                continue
            img_path = os.path.join(vrd_img_dir, img_name)
            shutil.copy(img_path, JPEGImages_path)
    print('Image total: %d' % len(os.listdir(JPEGImages_path)))
    print('Preparing JPEGImages done.')


def gen_ImageSets(vrd_root, vrd_pascal_root):
    # output path
    ImageSets_path = os.path.join(vrd_pascal_root, 'ImageSets', 'Main')
    if not os.path.isdir(ImageSets_path):
        os.makedirs(ImageSets_path)

    # original path
    train_img_root = os.path.join(vrd_root, 'sg_dataset', 'sg_train_images')
    test_img_root = os.path.join(vrd_root, 'sg_dataset', 'sg_test_images')
    img_dirs = {'train': train_img_root, 'test': test_img_root}

    # generate lists
    datasets = {}
    for (ds_name, img_root) in img_dirs.items():
        org_img_names = os.listdir(img_root)
        # ATTENTION: only JPEG image is legal
        img_names = filter(lambda id: id.endswith('jpg'), org_img_names)
        datasets[ds_name] = img_names

    # split 200 images from trainset as valset
    datasets['trainval'] = datasets['train']
    datasets['val'] = datasets['trainval'][-200:]
    datasets['train'] = datasets['trainval'][:-200]

    # save
    for ds in datasets:
        list_path = os.path.join(ImageSets_path, ds)+'.txt'
        with open(list_path, 'w') as f:
            f.writelines([s+'\n' for s in datasets[ds]])
    print('Preparing ImageSets done.')


def gen_Annotations(vrd_root, vrd_pascal_root):
    Annotations_path = os.path.join(vrd_pascal_root, 'Annotations')
    JPEGImages_path = os.path.join(vrd_pascal_root, 'JPEGImages')
    if not os.path.isdir(Annotations_path):
        os.makedirs(Annotations_path)

    # object category list
    obj_label_list_path = os.path.join(vrd_root, 'object_labels.txt')
    with open(obj_label_list_path, 'r') as f:
        raw_obj_labels = f.readlines()
        obj_labels = [s.strip() for s in raw_obj_labels]

    # original annotation file
    vrd_json_root = os.path.join(vrd_root, 'json_dataset')
    for ds_json_name in os.listdir(vrd_json_root):  # train / test
        ds_json_path = os.path.join(vrd_json_root, ds_json_name)
        ds_anno_pkg = json.load(open(ds_json_path))
        for i, (img_name, org_anno) in enumerate(ds_anno_pkg.items()):
            if (i+1) % 1000 == 0:
                print(i+1)
            if not img_name.endswith('jpg'):
                # ATTENTION: only JPEG image is legal
                print(img_name + ' is discarded.')
                continue
            # removing redundant objects from relations
            obj_label_boxes = []
            for rlt in org_anno:
                obj_sbj = [rlt['object'], rlt['subject']]
                for obj in obj_sbj:
                    # left top, right bottom
                    # ymin, ymax, xmin, xmax, category
                    label_box = obj['bbox']
                    label_box.append(obj['category'])
                    obj_label_boxes.append(label_box)

            objs = []
            # remove redundant objects
            if len(obj_label_boxes) > 0:
                obj_label_boxes = np.array(obj_label_boxes)
                unique_label_boxes = np.unique(obj_label_boxes, axis=0)
                for label_box in unique_label_boxes:
                    obj = dict()
                    obj['name'] = obj_labels[int(label_box[4])].strip()
                    obj['ymin'] = int(label_box[0])
                    obj['ymax'] = int(label_box[1])
                    obj['xmin'] = int(label_box[2])
                    obj['xmax'] = int(label_box[3])
                    obj['pose'] = 'Left'
                    obj['truncated'] = 0
                    obj['difficult'] = 0
                    objs.append(obj)

            img_path = os.path.join(JPEGImages_path, img_name)
            im = cv2.imread(img_path)
            if im is None:
                print(img_name + ' not found.')
                continue
            im_height = im.shape[0]
            im_width = im.shape[1]
            mid_anno = {'filename': img_name,
                        'width': im_width,
                        'height': im_height,
                        'depth': 3,
                        'objects': objs}
            pascal_anno_path = os.path.join(Annotations_path, img_name.split('.')[0]+'.xml')
            output_pascal_format(mid_anno, pascal_anno_path)
    print('Annotation total: %d' % len(os.listdir(Annotations_path)))
    print('Preparing Annotations done.')


if __name__ == '__main__':
    gen_JPEGImages(vrd_root, vrd_pascal_root)
    gen_ImageSets(vrd_root, vrd_pascal_root)
    gen_Annotations(vrd_root, vrd_pascal_root)