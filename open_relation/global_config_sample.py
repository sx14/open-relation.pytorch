import os


project_root = '/media/sunx/Data/linux-workspace/python-workspace/open-relation'


dataset_root = {
    'vg': os.path.join(project_root, 'data', 'VGdevkit2007', 'VOC2007'),
    'vrd': os.path.join(project_root, 'data', 'VRDdevkit2007', 'VOC2007')
}

our_model_root = os.path.join(project_root, 'open_relation', 'model')
