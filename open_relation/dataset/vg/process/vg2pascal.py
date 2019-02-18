import os
import json
from open_relation.dataset.lib.to_pascal_format import output_pascal_format
from open_relation.dataset.dataset_config import DatasetConfig

def convert_anno(org_anno):
    mid_anno = dict()
    mid_anno['filename'] = str(org_anno['image_info']['image_id'])+'.jpg'
    mid_anno['width'] = org_anno['image_info']['width']
    mid_anno['height'] = org_anno['image_info']['height']
    mid_anno['depth'] = 3
    mid_objects = []
    for org_obj in org_anno['objects']:
        mid_obj = dict()
        mid_obj['xmin'] = org_obj['xmin']
        mid_obj['xmax'] = org_obj['xmax']
        mid_obj['ymin'] = org_obj['ymin']
        mid_obj['ymax'] = org_obj['ymax']
        mid_obj['name'] = org_obj['name']
        mid_obj['pose'] = 'left'
        mid_obj['truncated'] = 0
        mid_obj['difficult'] = 0
        mid_objects.append(mid_obj)
    mid_anno['objects'] = mid_objects
    return mid_anno


def vg2pascal():
    vg_config = DatasetConfig('vg')
    json_anno_root = vg_config.data_config['clean_anno_root']
    pascal_anno_root = vg_config.pascal_format['Annotations']
    json_annos = os.listdir(json_anno_root)
    for i in range(len(json_annos)):
        print('processing vg2pascal: [%d/%d]' % (len(json_annos), i + 1))
        json_anno_path = os.path.join(json_anno_root, json_annos[i])
        json_anno = json.load(open(json_anno_path, 'r'))
        mid_anno = convert_anno(json_anno)
        pascal_anno_path = os.path.join(pascal_anno_root, json_annos[i][:-5]+'.xml')
        output_pascal_format(mid_anno, pascal_anno_path)