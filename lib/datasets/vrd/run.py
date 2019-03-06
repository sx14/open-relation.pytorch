import os
import shutil
import json
import pickle
import numpy as np
import cv2

from lib.datasets.vrd.label_hier.obj_hier import objnet
from lib.datasets.vrd.label_hier.pre_hier import prenet
from lib.datasets.tools.to_pascal_format import output_pascal_format
from lib.datasets.vrd.process.reformat_anno import reformat_anno
from lib.datasets.vrd.process.split_anno_pkg import split_anno_pkg
from global_config import PROJECT_ROOT


def prepare_relationship_roidb(objnet, prenet, anno_root, anno_list_path, box_label_path):
    # image id -> rlt info
    rlts = dict()

    # load img id list
    with open(anno_list_path, 'r') as anno_list_file:
        anno_list = anno_list_file.read().splitlines()

    # for each anno file
    for i in range(len(anno_list)):

        # load anno file
        anno_file_id = anno_list[i]
        print('prepare processing[%d/%d] %s' % (len(anno_list), (i + 1), anno_file_id))
        anno_path = os.path.join(anno_root, anno_list[i]+'.json')
        anno = json.load(open(anno_path, 'r'))
        image_id = anno_list[i]
        anno_rlts = anno['relationships']

        # collect boxes and labels
        rlt_info_list = []
        for rlt in anno_rlts:
            things = [rlt['predicate'], rlt['subject'], rlt['object']]
            labelnets = [prenet, objnet, objnet]
            # [ p_xmin, p_ymin, p_xmax, p_ymax, p_name,
            #   s_xmin, s_ymin, s_xmax, s_ymax, s_name,
            #   o_xmin, o_ymin, o_xmax, o_ymax, o_name  ]
            rlt_info = []
            # concatenate three box_label
            for j, thing in enumerate(things):
                xmin = int(thing['xmin'])
                ymin = int(thing['ymin'])
                xmax = int(thing['xmax'])
                ymax = int(thing['ymax'])
                label_ind = labelnets[j].get_node_by_name(thing['name']).index()
                rlt_info += [xmin, ymin, xmax, ymax, label_ind]
            rlt_info_list.append(rlt_info)
        rlts[image_id] = rlt_info_list
    with open(box_label_path, 'wb') as box_label_file:
        pickle.dump(rlts, box_label_file)


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
        img_names = filter(lambda id: os.path.exists(os.path.join(img_root, id)), img_names)
        img_names = map(lambda img_name: img_name.split('.')[0], img_names)
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
            # collect relationships
            obj_box_labels = []
            for rlt in org_anno:
                obj_sbj = [rlt['object'], rlt['subject']]
                for obj in obj_sbj:
                    # left top, right bottom
                    # ymin, ymax, xmin, xmax, category
                    box_label = obj['bbox']
                    box_label.append(obj['category'])
                    obj_box_labels.append(box_label)

            objs = []
            # remove redundant objects
            if len(obj_box_labels) > 0:
                obj_box_labels = np.array(obj_box_labels)
                unique_box_labels = np.unique(obj_box_labels, axis=0)
                for box_label in unique_box_labels:
                    obj = dict()
                    obj['name'] = obj_labels[int(box_label[4])].strip()
                    obj['ymin'] = int(box_label[0])
                    obj['ymax'] = int(box_label[1])
                    obj['xmin'] = int(box_label[2])
                    obj['xmax'] = int(box_label[3])
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
                        'objects': objs,
                        'relationships': org_anno}
            pascal_anno_path = os.path.join(Annotations_path, img_name.split('.')[0]+'.xml')
            output_pascal_format(mid_anno, pascal_anno_path)
    print('Annotation total: %d' % len(os.listdir(Annotations_path)))
    print('Preparing Annotations done.')


if __name__ == '__main__':
    vrd_root = label_path = os.path.join(PROJECT_ROOT, 'data', 'VRDdevkit2007', 'VOC2007')
    vrd_config = {
        'raw_anno_root': os.path.join(vrd_root, 'json_dataset'),
        'dirty_anno_root': os.path.join(vrd_root, 'dirty_anno'),
        'clean_anno_root': os.path.join(vrd_root, 'anno'),
        'obj_raw_label_path': os.path.join(vrd_root, 'object_labels.txt'),
        'pre_raw_label_path': os.path.join(vrd_root, 'predicate_labels.txt'),
        'Annotations': os.path.join(vrd_root, 'Annotations'),
        'ImageSets': os.path.join(vrd_root, 'ImageSets'),
    }

    # # to pascal format for object detection training
    # gen_JPEGImages(vrd_root, vrd_root)
    # gen_ImageSets(vrd_root, vrd_root)
    # gen_Annotations(vrd_root, vrd_root)
    #
    # # to json format for predicate recognition training
    # split_anno_pkg(vrd_config)
    # reformat_anno(vrd_config)

    # for eval
    roidb_save_path = os.path.join(PROJECT_ROOT, 'hier_rela', 'gt_box_label_vrd.bin')
    anno_root = vrd_config['clean_anno_root']
    anno_list_path = os.path.join(vrd_config['ImageSets'], 'test.txt')
    prepare_relationship_roidb(objnet, prenet, anno_root, anno_list_path, roidb_save_path)


