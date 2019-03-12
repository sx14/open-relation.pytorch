import os
from global_config import PROJECT_ROOT
from process.split_anno_pkg import split_anno_pkg
from process.vg2pascal import vg2pascal
from process.raw2wn import raw2wn

if __name__ == '__main__':
    vg_root = os.path.join(PROJECT_ROOT, 'data', 'VGdevkit2007', 'VOC2007')
    vg_config = {
        'vts_anno_path': os.path.join(vg_root, 'vg1_2_meta.h5'),
        'raw_anno_root': os.path.join(vg_root, 'json_dataset'),
        'dirty_anno_root': os.path.join(vg_root, 'dirty_anno'),
        'clean_anno_root': os.path.join(vg_root, 'anno'),
        'obj_raw_label_path': os.path.join(vg_root, 'object_labels.txt'),
        'obj_raw2wn_path': os.path.join(vg_root, 'object_label2wn.txt'),
        'pre_raw_label_path': os.path.join(vg_root, 'predicate_labels.txt'),
        'pre_raw2wn_path': os.path.join(vg_root, 'predicate_label2wn.txt'),
        'Annotations': os.path.join(vg_root, 'Annotations'),
        'ImageSets': os.path.join(vg_root, 'ImageSets', 'Main'),
        'JPEGImages': os.path.join(vg_root, 'JPEGImages'),
        'ds_root': vg_root
    }



    split_anno_pkg(vg_config)
    vg2pascal(vg_config)
    raw2wn(vg_config['obj_raw_label_path'], vg_config['obj_raw2wn_path'])

    # TODO: predicate part
    pass