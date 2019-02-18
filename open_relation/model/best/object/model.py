import h5py
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from open_relation.model.order_func import order_sim


class HypernymVisual(nn.Module):
    def __init__(self, vfeature_d, hidden_d, embedding_d, label_vec_path):
        super(HypernymVisual, self).__init__()
        self.hidden_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(vfeature_d, hidden_d)
        )

        self.embedding_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_d, embedding_d)
        )

        label_vec_file = h5py.File(label_vec_path, 'r')
        gt_label_vecs = np.array(label_vec_file['label_vec'])
        self._gt_label_vecs = Variable(torch.from_numpy(gt_label_vecs)).float().cuda()

    def forward(self, vfs):
        vf_hidden = self.hidden_layer(vfs)
        vf_embeddings = self.embedding_layer(vf_hidden)
        score_stack = Variable(torch.zeros(len(vf_embeddings), len(self._gt_label_vecs))).cuda()
        for i in range(len(vf_embeddings)):
            order_sims = order_sim(self._gt_label_vecs, vf_embeddings[i])
            score_stack[i] = order_sims
        return score_stack, vf_embeddings

    def embedding(self, vfs):
        vf_hidden = self.hidden_layer(vfs)
        vf_embeddings = self.embedding_layer(vf_hidden)
        return vf_embeddings