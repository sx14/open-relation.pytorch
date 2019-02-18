import os
import random
from open_relation.dataset.dataset_config import DatasetConfig


def split_dataset():
    vg_config = DatasetConfig('vg')
    anno_root = vg_config.data_config['clean_anno_root']
    anno_list = os.listdir(anno_root)
    anno_sum = len(anno_list)
    print('>>> Split dataset: image num = %d' % anno_sum)
    # train : test = 4 : 1
    # test_capacity = anno_sum / 5

    test_capacity = 5000
    val_capacity = 400
    train_capacity = anno_sum - val_capacity - test_capacity

    # random.shuffle(anno_list)
    split_list = {
        'trainval'  : anno_list[:train_capacity+val_capacity],
        'train'     : anno_list[:train_capacity],
        'val'       : anno_list[train_capacity:train_capacity+val_capacity],
        'test'      : anno_list[train_capacity+val_capacity:train_capacity+val_capacity+test_capacity],
    }

    # save split list
    split_list_root = vg_config.pascal_format['ImageSets']
    for d in split_list:
        image_id_list = []
        lines = split_list[d]

        for l in lines:
            image_id_list.append(l.split('.')[0]+'\n')
        list_file_path = os.path.join(split_list_root, d+'.txt')

        with open(list_file_path, 'w') as list_file:
            list_file.writelines(image_id_list)