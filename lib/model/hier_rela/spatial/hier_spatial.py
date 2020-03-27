import torch
from torch import nn
import h5py
import numpy as np
from torch.autograd import Variable

from lib.model.hier_utils.order_similarity import OrderSimilarity
from model.hier_utils.order_loss import OrderLoss


class HierSpatial(nn.Module):
    def __init__(self, prenet, label_vec_path):
        super(HierSpatial, self).__init__()

        # label vectors
        with h5py.File(label_vec_path, 'r') as f:
            label_vecs = np.array(f['label_vec'])

        with torch.no_grad():
            self.label_vecs = Variable(torch.from_numpy(label_vecs).float()).cuda()

        # visual embedding vector length
        self.embedding_len = self.label_vecs.size(1)
        self._loss = OrderLoss(prenet)
        # (batch,64,64,2)->(batch,60,60,64)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=5)
        # (batch,60,60,64)->(batch,30,30,64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (batch,30,30,64)->(batch,26,26,32)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5)
        # (batch,26,26,32)->(batch,13,13,32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.spa_cls_score = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(5408, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, self.embedding_len)
        )
        self.order_score = OrderSimilarity(norm=2)

    def forward(self, spa_maps, pre_labels):
        batch_size = 1
        conv1 = self.conv1(spa_maps)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        pool_feat = pool2.view(spa_maps.shape[0], -1)
        pre_embedding = self.spa_cls_score(pool_feat)
        # compute order similarity
        if pre_embedding.size(0) < 30:
            # fast, memory consuming
            cls_score_use = self.order_score.forward(self.label_vecs, pre_embedding)
        else:
            # slow, memory saving
            cls_score_use = self.order_score.forward1(self.label_vecs, pre_embedding)

        RCNN_loss_cls = 0

        if self.training:
            RCNN_loss_cls = self._loss.forward(cls_score_use, pre_labels)

        cls_score = cls_score_use.view(batch_size, spa_maps.size()[0], -1)

        return cls_score, RCNN_loss_cls
