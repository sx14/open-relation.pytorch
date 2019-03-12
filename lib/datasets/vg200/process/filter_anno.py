import os
import json
from lib.datasets.vg1000.label_hier.obj_hier import objnet
from lib.datasets.vg1000.label_hier.pre_hier import prenet


def filter_anno(vg_config):
    dirty_anno_root = vg_config['dirty_anno_root']
    clean_anno_root = vg_config['clean_anno_root']

    if os.path.exists(clean_anno_root):
        import shutil
        shutil.rmtree(clean_anno_root)
        os.makedirs(clean_anno_root)

    anno_list = os.listdir(dirty_anno_root)
    anno_num = len(anno_list)

    obj_raw_labels = set(objnet.get_raw_labels())
    pre_raw_labels = set(prenet.get_raw_labels())

    for i in range(anno_num):
        print('filtering [%d/%d]' % (anno_num, i+1))
        anno_name = anno_list[i]

        # load dirty json anno
        dirty_anno_path = os.path.join(dirty_anno_root, anno_name)
        dirty_anno = json.load(open(dirty_anno_path, 'r'))

        # keep objects in label set
        clean_objects = []
        dirty_objects = dirty_anno['objects']
        for d_obj in dirty_objects:
            if d_obj['name'] in obj_raw_labels:
                clean_objects.append(d_obj)

        # keep relationships whose sbj,obj,pre are in label set
        clean_relations = []
        dirty_relations = dirty_anno['relationships']
        for d_rlt in dirty_relations:
            keep_rlt = True
            r_objs = [d_rlt['subject'], d_rlt['object']]

            if d_rlt['predicate']['name'] not in pre_raw_labels:
                keep_rlt = False
                continue

            for r_obj in r_objs:
                if r_obj['name'] not in obj_raw_labels:
                    keep_rlt = False
                    break

            if keep_rlt:
                clean_relations.append(d_rlt)

        if len(clean_objects) == 0 or len(clean_relations) == 0:
            continue

        # save cleaned json anno
        clean_anno = dict()
        clean_anno['objects'] = clean_objects
        clean_anno['relationships'] = clean_relations
        clean_anno['image_info'] = dirty_anno['image_info']

        clean_anno_path = os.path.join(clean_anno_root, anno_name)
        json.dump(clean_anno, open(clean_anno_path, 'w'))

    clean_annos = os.listdir(clean_anno_root)
    print('>>> filter_anno: image num = %d' % (len(clean_annos)))




