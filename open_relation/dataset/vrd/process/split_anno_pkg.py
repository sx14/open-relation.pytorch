import os
import json
import shutil
from open_relation.dataset.dataset_config import DatasetConfig


def split_anno_pkg():
    data_config = DatasetConfig('vrd')
    # ====== split annotation package ======
    datasets = ['train', 'test']
    # contain image lists
    dataset_lists = {'train': [], 'test': []}
    list_root = data_config.pascal_format['ImageSets']

    # all images and annotations are saved together
    image_root = data_config.pascal_format['JPEGImages']
    splited_anno_root = data_config.data_config['dirty_anno_root']


    for d in datasets:
        anno_package_path = os.path.join(data_config.dataset_root, 'json_dataset', 'annotations_' + d + '.json')
        anno_package = json.load(open(anno_package_path))

        data_list = dataset_lists[d]
        d_image_root = os.path.join(data_config.dataset_root, 'sg_dataset', 'sg_' + d + '_images')

        for i, img_name in enumerate(anno_package.keys()):
            print('processing [%d/%d]' % (len(anno_package), i+1))
            anno = anno_package[img_name]

            # copy image
            # only jpeg image
            img_name = img_name.split('.')[0]+'.jpg'
            org_img_path = os.path.join(d_image_root, img_name)
            if not os.path.exists(org_img_path):
                continue
            dst_img_root = os.path.join(image_root)
            shutil.copy(org_img_path, dst_img_root)

            # record image name in list
            data_list.append(img_name.split('.')[0]+'\n')

            # save splited annotation
            anno_name = img_name.split('.')[0]+'.json'
            anno_save_path = os.path.join(splited_anno_root, anno_name)
            json.dump(anno, open(anno_save_path, 'w'))

        # save image list
        list_file_path = os.path.join(list_root, d+'.txt')
        list_file = open(list_file_path, 'w')
        list_file.writelines(data_list)
        list_file.close()


