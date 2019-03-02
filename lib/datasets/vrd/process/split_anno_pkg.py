import os
import json


def split_anno_pkg(data_config):
    # ====== split annotation package ======
    datasets = ['train', 'test']

    # all images and annotations are saved together
    splited_anno_root = data_config['dirty_anno_root']

    for d in datasets:
        anno_package_path = os.path.join(data_config['raw_anno_root'], 'annotations_' + d + '.json')
        anno_package = json.load(open(anno_package_path))

        for i, img_name in enumerate(anno_package.keys()):
            print('processing [%d/%d]' % (len(anno_package), i+1))
            anno = anno_package[img_name]

            # save splited annotation
            anno_name = img_name.split('.')[0]+'.json'
            anno_save_path = os.path.join(splited_anno_root, anno_name)
            json.dump(anno, open(anno_save_path, 'w'))


