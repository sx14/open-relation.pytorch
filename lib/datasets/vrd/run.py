import os
import shutil
import json
import pickle
import numpy as np
import cv2


from lib.datasets.tools.to_pascal_format import output_pascal_format
from lib.datasets.vrd.process.reformat_anno import reformat_anno
from lib.datasets.vrd.process.split_anno_pkg import split_anno_pkg
from global_config import PROJECT_ROOT, VRD_ROOT


def output_pre_freq(pre_count, save_path):
    lines = []
    N_rlt = sum(pre_count.values())
    for pre in pre_count:
        N_pre = pre_count[pre]
        line = '%s|%.4f\n' % (pre, N_pre / float(N_rlt))
        lines.append(line)

    with open(save_path, 'w') as f:
        f.writelines(lines)


def all_relationships(anno_root, anno_list_path):
    # sbj -> obj -> pre
    rlts = dict()
    pre_count = dict()

    # load img id list
    with open(anno_list_path, 'r') as anno_list_file:
        anno_list = anno_list_file.read().splitlines()

    # for each anno file
    for i in range(len(anno_list)):

        # load anno file
        anno_file_id = anno_list[i]
        print('look reltionships [%d/%d] %s' % (len(anno_list), (i + 1), anno_file_id))
        anno_path = os.path.join(anno_root, anno_list[i]+'.json')
        anno = json.load(open(anno_path, 'r'))
        image_id = anno_list[i]
        anno_rlts = anno['relationships']
        if len(anno_rlts) == 0:
            continue

        # collect boxes and labels
        rlt_info_list = []

        for rlt in anno_rlts:
            sbj = rlt['subject']
            obj = rlt['object']
            pre = rlt['predicate']

            if sbj['name'] not in rlts:
                rlts[sbj['name']] = {}

            obj2pre = rlts[sbj['name']]
            if obj['name'] not in obj2pre:
                obj2pre[obj['name']] = set()

            pres = obj2pre[obj['name']]
            pres.add(pre['name'])

            if pre['name'] not in pre_count:
                pre_count[pre['name']] = 0

            pre_count[pre['name']] = pre_count[pre['name']] + 1

    return rlts, pre_count




def prepare_relationship_roidb(objnet, prenet, anno_root, anno_list_path, box_label_path, train_rlts):
    # image id -> rlt info
    rlts = dict()
    zero_count = 0

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
        if len(anno_rlts) == 0:
            continue

        # collect boxes and labels
        rlt_info_list = []

        for rlt in anno_rlts:
            sbj = rlt['subject']
            obj = rlt['object']
            pre = rlt['predicate']

            zero_shot = 1
            if sbj['name'] in train_rlts:
                obj2pre = train_rlts[sbj['name']]
                if obj['name'] in obj2pre:
                    pres = obj2pre[obj['name']]
                    if pre['name'] in pres:
                        # not zero shot
                        zero_shot = 0
            count = 0
            pair1 = np.concatenate((np.array(sbj), np.array(obj)), axis=1)
            for rlt1 in anno_rlts:
                sbj1 = rlt1['subject']
                obj1 = rlt1['object']
                pair2 = np.concatenate((np.array(sbj), np.array(obj)), axis=1)
                if np.sum(pair1 - pair2) == 0:
                    count += 1


            things = [pre, sbj, obj]
            labelnets = [prenet, objnet, objnet]
            # [ p_xmin, p_ymin, p_xmax, p_ymax, p_name,
            #   s_xmin, s_ymin, s_xmax, s_ymax, s_name,
            #   o_xmin, o_ymin, o_xmax, o_ymax, o_name,
            #   p_conf, s_conf, o_conf, is_zero]
            rlt_info = []

            # concatenate three box_label
            for j, thing in enumerate(things):
                xmin = int(thing['xmin'])
                ymin = int(thing['ymin'])
                xmax = int(thing['xmax'])
                ymax = int(thing['ymax'])
                label_ind = labelnets[j].get_node_by_name(thing['name']).index()
                rlt_info += [xmin, ymin, xmax, ymax, label_ind]
            rlt_info += [1.0, 1.0, 1.0, zero_shot, count]
            zero_count += zero_shot
            rlt_info_list.append(rlt_info)
        rlts[image_id] = rlt_info_list
    print('zero: %d' % zero_count)
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
            # if not img_name.endswith('jpg'):
                # ATTENTION: only JPEG image is legal
                # print(img_name + ' is discarded.')
                # continue
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
    vrd_config = {
        'pre_freq_path': os.path.join(VRD_ROOT, 'pre_freq.txt'),
        'raw_anno_root': os.path.join(VRD_ROOT, 'json_dataset'),
        'dirty_anno_root': os.path.join(VRD_ROOT, 'dirty_anno'),
        'clean_anno_root': os.path.join(VRD_ROOT, 'anno'),
        'obj_raw_label_path': os.path.join(VRD_ROOT, 'object_labels.txt'),
        'pre_raw_label_path': os.path.join(VRD_ROOT, 'predicate_labels.txt'),
        'Annotations': os.path.join(VRD_ROOT, 'Annotations'),
        'ImageSets': os.path.join(VRD_ROOT, 'ImageSets'),
    }

    # # to pascal format for object detection training
    # gen_JPEGImages(VRD_ROOT, VRD_ROOT)
    # gen_ImageSets(VRD_ROOT, VRD_ROOT)
    # gen_Annotations(VRD_ROOT, VRD_ROOT)
    #
    # # to json format for predicate recognition training
    split_anno_pkg(vrd_config)
    reformat_anno(vrd_config)

    # for eval
    roidb_save_path = os.path.join(PROJECT_ROOT, 'hier_rela', 'gt_rela_roidb_vrd.bin')
    anno_root = vrd_config['clean_anno_root']

    train_anno_list_path = os.path.join(vrd_config['ImageSets'], 'Main', 'trainval.txt')
    train_rlts, train_pre_counts = all_relationships(anno_root, train_anno_list_path)

    output_pre_freq(train_pre_counts, vrd_config['pre_freq_path'])

    test_anno_list_path = os.path.join(vrd_config['ImageSets'], 'Main', 'test.txt')
    from lib.datasets.vrd.label_hier.obj_hier import objnet
    from lib.datasets.vrd.label_hier.pre_hier import prenet
    prepare_relationship_roidb(objnet, prenet, anno_root, test_anno_list_path, roidb_save_path, train_rlts)


