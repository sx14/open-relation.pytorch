import os
from open_relation import global_config
from open_relation.dataset.dataset_config import DatasetConfig

log_root = 'open_relation/log'

vg_dataset_config = DatasetConfig('vg')

vg_obj_hyper_params = {
    'visual_d': 4096,
    'hidden_d': 4096,
    'embedding_d': 600,
    'epoch': 20,
    'batch_size': 64,
    'negative_label_num': 2450,
    'eval_freq': 5000,
    'print_freq': 10,
    'lr': 0.01,
    'visual_feature_root': vg_dataset_config.extra_config['object'].fc7_root,
    'list_root': vg_dataset_config.extra_config['object'].label_root,
    'raw2weight_path': vg_dataset_config.extra_config['object'].config['raw2weight_path'],
    'label_vec_path': os.path.join(global_config.project_root, 'open_relation', 'label_embedding', 'object', 'label_vec_vg.h5'),
    'label_vec_path1': os.path.join(global_config.project_root, 'open_relation', 'label_embedding', 'object', 'label_vec_vg1.h5'),
    'latest_weight_path': os.path.join(global_config.our_model_root, 'object', 'vg_weights.pkl'),
    'best_weight_path': os.path.join(global_config.our_model_root, 'object', 'vg_weights_best.pkl'),
    'eval_weight_path':os.path.join(global_config.our_model_root, 'best', 'vg_obj_weights.pkl'),
}

vg_pre_hyper_params = {
    'visual_d': 4096*3,
    'hidden_d': 4096,
    'embedding_d': 600,
    'epoch': 200,
    'batch_size': 256,
    'negative_label_num': 78,
    'eval_freq': 500,
    'print_freq': 10,
    'lr': 0.01,
    'visual_feature_root': vg_dataset_config.extra_config['predicate'].fc7_root,
    'list_root': vg_dataset_config.extra_config['predicate'].label_root,
    'raw2weight_path': vg_dataset_config.extra_config['predicate'].config['raw2weight_path'],
    'label_vec_path': os.path.join(global_config.project_root, 'open_relation', 'label_embedding', 'predicate', 'label_vec_vg.h5'),
    'label_vec_path1': os.path.join(global_config.project_root, 'open_relation', 'label_embedding', 'predicate', 'label_vec_vg1.h5'),
    'latest_weight_path': os.path.join(global_config.our_model_root, 'predicate', 'vg_weights.pkl'),
    'best_weight_path': os.path.join(global_config.our_model_root, 'predicate', 'vg_weights_best.pkl'),
    'eval_weight_path':os.path.join(global_config.our_model_root, 'best', 'vg_pre_weights.pkl'),
}


vrd_dataset_config = DatasetConfig('vrd')

vrd_obj_hyper_params = {
    'visual_d': 4096,
    'hidden_d': 4096,
    'embedding_d': 600,
    'epoch': 200,
    'batch_size': 64,
    'negative_label_num': 330,
    'eval_freq': 1000,
    'print_freq': 10,
    'lr': 0.01,
    'visual_feature_root': vrd_dataset_config.extra_config['object'].fc7_root,
    'list_root': vrd_dataset_config.extra_config['object'].label_root,
    'raw2weight_path': vrd_dataset_config.extra_config['object'].config['raw2weight_path'],
    'label_vec_path': os.path.join(global_config.project_root, 'open_relation', 'label_embedding', 'object', 'label_vec_vrd.h5'),
    'label_vec_path1': os.path.join(global_config.project_root, 'open_relation', 'label_embedding', 'object', 'label_vec_vrd1.h5'),
    'latest_weight_path': os.path.join(global_config.our_model_root, 'object', 'vrd_weights.pkl'),
    'best_weight_path': os.path.join(global_config.our_model_root, 'object', 'vrd_weights_best.pkl'),
    'eval_weight_path':os.path.join(global_config.our_model_root, 'best', 'vrd_obj_weights.pkl'),
}

vrd_pre_hyper_params = {
    'visual_d': 4096*3,
    'hidden_d': 4096,
    'embedding_d': 600,
    'epoch': 200,
    'batch_size': 64,
    'negative_label_num': 78,
    'eval_freq': 500,
    'print_freq': 10,
    'lr': 0.01,
    'visual_feature_root': vrd_dataset_config.extra_config['predicate'].fc7_root,
    'list_root': vrd_dataset_config.extra_config['predicate'].label_root,
    'raw2weight_path': vrd_dataset_config.extra_config['predicate'].config['raw2weight_path'],
    'label_vec_path': os.path.join(global_config.project_root, 'open_relation', 'label_embedding', 'predicate', 'label_vec_vrd.h5'),
    'label_vec_path1': os.path.join(global_config.project_root, 'open_relation', 'label_embedding', 'predicate', 'label_vec_vrd1.h5'),
    'latest_weight_path': os.path.join(global_config.our_model_root, 'predicate', 'vrd_weights.pkl'),
    'best_weight_path': os.path.join(global_config.our_model_root, 'predicate', 'vrd_weights_best.pkl'),
    'eval_weight_path':os.path.join(global_config.our_model_root, 'best', 'vrd_obj_weights.pkl'),
}

hyper_params = {
    'vg': {'predicate': vg_pre_hyper_params,
            'object': vg_obj_hyper_params},
    'vrd': {'predicate': vrd_pre_hyper_params,
            'object': vrd_obj_hyper_params}
}