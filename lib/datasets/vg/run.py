import os

from process.split_anno_pkg import split_anno_pkg
from process.reformat_anno import reformat_anno
from process.collect_labels import collect_labels
from process.filter_anno import filter_anno
from process.split_dataset import split_dataset
from process.vg2pascal import vg2pascal
from global_config import PROJECT_ROOT

if __name__ == '__main__':
    vg_root = os.path.join(PROJECT_ROOT, 'data', 'VGdevkit2007', 'VOC2007')
    vg_config = {
        'raw_anno_root': os.path.join(vg_root, 'json_dataset'),
        'dirty_anno_root': os.path.join(vg_root, 'dirty_anno'),
        'obj_raw_label_path': os.path.join(vg_root, 'object_labels.txt'),
        'obj_raw2wn_path': os.path.join(vg_root, 'object_label2wn.txt'),
        'pre_raw_label_path': os.path.join(vg_root, 'predicate_labels.txt'),
        'pre_raw2wn_path': os.path.join(vg_root, 'predicate_label2wn.txt'),
        'Annotations': os.path.join(vg_root, 'Annotations'),
        'ImageSets': os.path.join(vg_root, 'ImageSets'),
    }

    split_anno_pkg(vg_config)
    reformat_anno(vg_config)
    collect_labels(vg_config)
    filter_anno(vg_config)
    split_dataset(vg_config)
    vg2pascal(vg_config)
    # TODO: predicate part
    pass