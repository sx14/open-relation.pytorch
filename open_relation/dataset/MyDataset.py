import os
import random
import pickle
import numpy as np
import h5py
import torch


class MyDataset():
    def __init__(self, raw_feature_root, flabel_list_path, raw2path, vis_feat_dim,
                 raw2weight_path, label_num, minibatch_size=64, negative_label_num=50):
        # whole dataset
        self._label_num = label_num
        self.vis_feat_dim = vis_feat_dim
        self._minibatch_size = minibatch_size
        self._negative_label_num = negative_label_num
        self._raw_feature_root = raw_feature_root

        self._feature_indexes = []      # index feature: [feature file name, offset]
        self._label_indexes = []        # label index
        # cached feature package
        self._curr_package = dict()
        # number of image_feature file
        self._curr_package_capacity = 5000
        # package bounds
        self._curr_package_start_fid = 0
        self._next_package_start_fid = 0
        # _curr_package_cursor indexes _curr_package_feature_indexes
        self._curr_package_cursor = -1
        # random current package feature indexes of the whole feature list
        self._curr_package_feature_indexes = []
        # label2path
        self._raw2path = raw2path
        self._raw2weight = pickle.load(open(raw2weight_path, 'rb'))

        # roidb
        with open(flabel_list_path, 'r') as list_file:
            flabel_list = list_file.read().splitlines()
        feat_file_names = set()
        for item in flabel_list:
            # image id, offset, hier_label_index, vg_label_index
            item_info = item.split(' ')
            item_feature_file = item_info[0]
            feat_file_names.add(item_feature_file)
            item_id = int(item_info[1])
            item_label_index = int(item_info[2])
            item_vg_index = int(item_info[3])
            # label indexes [hier_label_index, vg_label_index]
            self._label_indexes.append([item_label_index, item_vg_index])
            # feature indexes [feature file name, offset]
            self._feature_indexes.append([item_feature_file, item_id])
        self._raw_feature_file_num = len(feat_file_names)

    def init_package(self):
        if self._curr_package_capacity < self._raw_feature_file_num:
            self._next_package_start_fid = 0
            self._curr_package_start_fid = 0
            self._curr_package_feature_indexes = []
            self._curr_package_cursor = 0
        else:
            self._curr_package_cursor = 0
            random.shuffle(self._curr_package_feature_indexes)

    def __len__(self):
        return len(self._feature_indexes)

    def has_next_minibatch(self):
        if self._next_package_start_fid == len(self._feature_indexes):
            # the last package
            if self._curr_package_cursor == len(self._curr_package_feature_indexes):
                # the last minibatch
                return False
        return True

    def load_next_feature_package(self):
        if self._next_package_start_fid == len(self._feature_indexes):
            # No more package
            return

        print('Loading features into memory ......')
        del self._curr_package          # release memory
        self._curr_package = dict()     # feature_file -> [f1,f2,f3,...]
        self._curr_package_start_fid = self._next_package_start_fid
        while len(self._curr_package.keys()) < self._curr_package_capacity:
            if self._next_package_start_fid == len(self._feature_indexes):
                # all features already loaded
                break
            # fill feature package
            next_feature_file, _ = self._feature_indexes[self._next_package_start_fid]
            if next_feature_file not in self._curr_package.keys():
                feature_path = os.path.join(self._raw_feature_root, next_feature_file)
                with open(feature_path, 'rb') as feature_file:
                    features = pickle.load(feature_file)
                    self._curr_package[next_feature_file] = features
            self._next_package_start_fid += 1
        self._curr_package_feature_indexes = np.arange(self._curr_package_start_fid, self._next_package_start_fid)
        # shuffle the feature indexes of current feature package
        random.shuffle(self._curr_package_feature_indexes)
        # init package index cursor
        self._curr_package_cursor = 0

    def minibatch(self):
        vfs = np.zeros((self._minibatch_size, self.vis_feat_dim))
        p_n_ls = np.zeros((self._minibatch_size, self._negative_label_num+1)).astype(np.int)
        pws = np.zeros(self._minibatch_size)
        v_actual_num = 0
        for v in range(0, self._minibatch_size):
            if self._curr_package_cursor == len(self._curr_package_feature_indexes):
                # current package finished, load another 4000 feature files
                self.load_next_feature_package()

            if self._curr_package_cursor == len(self._curr_package_feature_indexes):
                vfs = vfs[:v_actual_num]
                p_n_ls = p_n_ls[:v_actual_num]
                pws = pws[:v_actual_num]
                break

            fid = self._curr_package_feature_indexes[self._curr_package_cursor]
            feature_file, offset = self._feature_indexes[fid]
            vfs[v] = self._curr_package[feature_file][offset]
            all_nls = list(set(range(self._label_num)) - set(self._raw2path[self._label_indexes[fid][1]]))
            p_n_ls[v] = [self._label_indexes[fid][0]] + random.sample(all_nls, self._negative_label_num)
            pws[v] = self._raw2weight[self._label_indexes[fid][1]]
            self._curr_package_cursor += 1
            v_actual_num += 1

        #  vfs: minibatch_size | pls: minibatch_size | nls: minibatch_size
        vfs = torch.from_numpy(vfs).float()
        p_n_ls = torch.from_numpy(p_n_ls)
        pws = torch.from_numpy(pws).float()
        # Tensor to Variable
        vfs = torch.autograd.Variable(vfs).cuda()
        pws = torch.autograd.Variable(pws).cuda()
        return vfs, p_n_ls, pws




