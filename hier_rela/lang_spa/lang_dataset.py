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

    def __init__(self, rlt_path, obj_label_vec_path, pre_label_vec_path, prenet):
        obj_vec_file = h5py.File(obj_label_vec_path, 'r')
        self._obj_vecs = torch.from_numpy(np.array(obj_vec_file['label_vec']))

        pre_vec_file = h5py.File(pre_label_vec_path, 'r')
        self._pre_vecs = torch.from_numpy(np.array(pre_vec_file['label_vec']))

        rlts = np.load(rlt_path+'.npy')
        self._rlts = np.array(rlts)

        self._raw2path = prenet.raw2path()
        self._label_sum = prenet.label_sum()
        self._neg_sample_num = prenet.neg_num()

    def __getitem__(self, item):
        rlt = self._rlts[item]
        pre_vec = self._pre_vecs[rlt[4]]
        sbj_vec = self._obj_vecs[rlt[9]]
        obj_vec = self._obj_vecs[rlt[14]]
        pos_inds = self._raw2path[rlt[4]]

        all_neg_inds = list(set(range(self._label_sum)) - set(pos_inds))
        neg_inds = random.sample(all_neg_inds, self._neg_sample_num)
        pos_neg_inds = np.array([rlt[4]] + neg_inds)
        rlt = np.array(rlt)

        return [sbj_vec, pre_vec, obj_vec, pos_neg_inds, rlt]

    def __len__(self):
        return len(self._rlts)