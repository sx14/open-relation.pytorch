import h5py
import numpy as np
import random
import torch
from torch.utils.data import Dataset


class LangDataset(Dataset):

    def obj_vec_length(self):
        return self._obj_vecs.shape[1]

    def get_gt_vecs(self):
        return self._pre_vecs

    def __init__(self, rlt_path, obj_label_vec_path, pre_label_vec_path):
        obj_vec_file = h5py.File(obj_label_vec_path, 'r')
        self._obj_vecs = torch.from_numpy(np.array(obj_vec_file['label_vec']))

        pre_vec_file = h5py.File(pre_label_vec_path, 'r')
        self._pre_vecs = torch.from_numpy(np.array(pre_vec_file['label_vec']))

        rlts = np.load(rlt_path+'.npy')
        self._rlts = np.array(rlts)

    def __getitem__(self, item):
        rlt = self._rlts[item]
        pre_vec = self._pre_vecs[rlt[4]]
        sbj_vec = self._obj_vecs[rlt[9]]
        obj_vec = self._obj_vecs[rlt[14]]
        rlt = np.array(rlt)
        y = 1
        if rlt[4] == 0:
            y = 0

        return [sbj_vec, pre_vec, obj_vec, rlt, y]

    def __len__(self):
        return len(self._rlts)
