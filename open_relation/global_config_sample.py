import os


project_root = '/media/sunx/Data/linux-workspace/python-workspace/hierarchical-relationship'


dataset_root = {
    'vg': os.path.join(project_root, 'VG'),
    'vrd': os.path.join(project_root, 'VRD')
}


fast_prototxt_path = os.path.join(project_root, 'models', 'VGG16', 'test.prototxt')
fast_caffemodel_path = os.path.join(project_root, 'data', 'fast_rcnn_models', 'vgg16_fast_rcnn_iter_40000.caffemodel')

our_model_root = os.path.join(project_root, 'open_relation', 'model')
