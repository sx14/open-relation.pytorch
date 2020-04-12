import os
import json
import pickle
from global_config import PROJECT_ROOT, VG_ROOT
from lib.datasets.vglsj.process.split_anno_pkg_pascal import split_anno_pkg
from lib.datasets.vglsj.process.vg2pascal import vg2pascal


if __name__ == '__main__':
    vg_config = {
        'vts_anno_path': os.path.join(VG_ROOT, 'vg_lsj_det.h5'),
        'raw_anno_root': os.path.join(VG_ROOT, 'json_dataset'),
        'dirty_anno_root': os.path.join(VG_ROOT, 'dirty_anno'),
        'clean_anno_root': os.path.join(VG_ROOT, 'anno_det'),
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
