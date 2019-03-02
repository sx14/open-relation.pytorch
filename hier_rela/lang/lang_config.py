import os


data_config = {
    'train': {
        'ext_rlt_path': 'train_ext_rlts',
        'raw_rlt_path': 'train_raw_rlts'},
    'test': {
        'ext_rlt_path': 'test_ext_rlts',
        'raw_rlt_path': 'test_raw_rlts',}
}

train_params = {
    'lr': 0.01,
    'epoch_num': 50,
    'batch_size': 128,
    'neg_sample_num': 80,
    'latest_model_path': 'output/lan_weights_new.pkl',
    'best_model_path': 'output/lan_weights_best.pkl',
    'save_model_path': 'output/lan_weights_'
}