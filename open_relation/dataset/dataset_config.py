import os
import open_relation.global_config


class ExtraConfig:
    def __init__(self, target, extra_root, dataset_name):
        self._root = os.path.join(extra_root, target)
        self.fc7_root = os.path.join(self._root, 'fc7')
        self.label_root = os.path.join(self._root, 'label')
        self.prepare_root = os.path.join(self._root, 'prepare')
        self.det_box_path = os.path.join(self._root, 'det', 'test_box.bin')
        self.config = {
            'raw_label_list': os.path.join(self.prepare_root, 'raw_labels.txt'),
            'label_vec_path': os.path.join(open_relation.global_config.project_root, 'open_relation',
                                           'label_embedding', target, 'label_vec_'+dataset_name+'.h5'),
            'raw2weight_path': os.path.join(self.prepare_root, 'raw2weight.bin')
        }


class DatasetConfig:
    def __init__(self, dataset_name):
        self.dataset_root = open_relation.global_config.dataset_root[dataset_name]
        self.extra_root = os.path.join(self.dataset_root, 'feature')

        self.data_config = {
            # dataset preprocess
            'img_root': os.path.join(self.dataset_root, 'JPEGImages'),
            'raw_anno_root': os.path.join(self.dataset_root, 'json_dataset'),
            'dirty_anno_root': os.path.join(self.dataset_root, 'dirty_anno'),
            'clean_anno_root': os.path.join(self.dataset_root, 'anno')
        }

        self.pascal_format = {
            'JPEGImages': os.path.join(self.dataset_root, 'JPEGImages'),
            'ImageSets': os.path.join(self.dataset_root, 'ImageSets', 'Main'),
            'Annotations': os.path.join(self.dataset_root, 'Annotations')
        }

        self.extra_config = {
            'object': ExtraConfig('object', self.extra_root, dataset_name),
            'predicate': ExtraConfig('predicate', self.extra_root, dataset_name)
        }




