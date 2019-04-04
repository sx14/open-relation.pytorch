import os
import json
import pickle
from global_config import PROJECT_ROOT, VG_ROOT
from process.split_anno_pkg import split_anno_pkg
from process.vg2pascal import vg2pascal
from process.raw2wn import raw2wn


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
            pair = [sbj, obj]
            for rlt1 in anno_rlts:
                sbj1 = rlt1['subject']
                obj1 = rlt1['object']
                pair1 = [sbj1, obj1]

                same = 1
                for p in range(2):
                    part = pair[p]
                    part1 = pair1[p]
                    if part['xmin'] == part1['xmin'] and \
                            part['ymin'] == part1['ymin'] and \
                            part['xmax'] == part1['xmax'] and \
                            part['ymax'] == part1['ymax']:
                        pass
                    else:
                        same = 0
                        break
                count += same

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



if __name__ == '__main__':
    vg_config = {
        'vts_anno_path': os.path.join(VG_ROOT, 'vg1_2_meta.h5'),
        'raw_anno_root': os.path.join(VG_ROOT, 'json_dataset'),
        'dirty_anno_root': os.path.join(VG_ROOT, 'dirty_anno'),
        'clean_anno_root': os.path.join(VG_ROOT, 'anno'),
        'obj_raw_label_path': os.path.join(VG_ROOT, 'object_labels.txt'),
        'obj_raw2wn_path': os.path.join(VG_ROOT, 'object_label2wn.txt'),
        'pre_raw_label_path': os.path.join(VG_ROOT, 'predicate_labels.txt'),
        'pre_raw2wn_path': os.path.join(VG_ROOT, 'predicate_label2wn.txt'),
        'Annotations': os.path.join(VG_ROOT, 'Annotations'),
        'ImageSets': os.path.join(VG_ROOT, 'ImageSets'),
        'JPEGImages': os.path.join(VG_ROOT, 'JPEGImages'),
        'pre_freq_path': os.path.join(VG_ROOT, 'pre_freq.txt'),
        'ds_root': VG_ROOT
    }

    split_anno_pkg(vg_config)
    vg2pascal(vg_config)
    raw2wn(vg_config['obj_raw_label_path'], vg_config['obj_raw2wn_path'])

    roidb_save_path = os.path.join(PROJECT_ROOT, 'hier_rela', 'gt_rela_roidb_vg.bin')
    anno_root = vg_config['clean_anno_root']

    train_anno_list_path = os.path.join(vg_config['ImageSets'], 'Main', 'trainval.txt')
    train_rlts, train_pre_counts = all_relationships(anno_root, train_anno_list_path)

    output_pre_freq(train_pre_counts, vg_config['pre_freq_path'])

    test_anno_list_path = os.path.join(vg_config['ImageSets'], 'Main', 'test.txt')
    from lib.datasets.vg200.label_hier.obj_hier import objnet
    from lib.datasets.vg200.label_hier.pre_hier import prenet

    prepare_relationship_roidb(objnet, prenet, anno_root, test_anno_list_path, roidb_save_path, train_rlts)
