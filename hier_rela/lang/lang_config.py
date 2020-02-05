import os
from global_config import PROJECT_ROOT

data_config = {
    'train': {
        'ext_rlt_path': 'train_ext_rlts_',
        'raw_rlt_path': 'train_raw_rlts_'},
    'test': {
        'ext_rlt_path': 'test_ext_rlts_',
        'raw_rlt_path': 'test_raw_rlts_', }
}

train_params = {
    'dataset': 'vrd',
    'lr': 0.01,
    'epoch_num': 150,
    'batch_size': 32,
    'latest_model_path': 'hier_rela_lan_new_',
    'best_model_path': 'hier_rela_lan_best_',
}

DATASET_ROOT = os.path.join(PROJECT_ROOT, 'hier_rela', 'lang', 'dataset')