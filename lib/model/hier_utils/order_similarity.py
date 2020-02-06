import torch
from torch import nn
from torch.autograd import Variable


class OrderSimilarity(nn.Module):
    def __init__(self, norm):
        super(OrderSimilarity, self).__init__()
        self._norm = norm
        self.act = nn.ReLU()

    def forward1(self, lab_vecs, vis_vecs):
        order_scores = Variable(torch.zeros(vis_vecs.size()[0], lab_vecs.size()[0])).cuda()
        for i in range(vis_vecs.size()[0]):
            # hyper - hypo
            sub = lab_vecs - vis_vecs[i]
            # max(sub, 0)
            sub = self.act(sub)
            # norm 2
            order_dis = sub.norm(p=self._norm, dim=1)
            # order_dis = order_dis + 0.00001
            # order_scores[i] = -torch.log(order_dis)
            order_sim = -order_dis
            order_scores[i] = order_sim

        return order_scores

    def forward(self, lab_vecs, vis_vecs):
        d_vec = lab_vecs.size(1)
        n_label = lab_vecs.size(0)
        n_vis = vis_vecs.size(0)

        stack_lab_vecs = lab_vecs.repeat(n_vis, 1)
        stack_vis_vecs = vis_vecs.repeat(1, n_label).reshape(n_vis * n_label, d_vec)

        stack_sub = stack_lab_vecs - stack_vis_vecs
        stack_sub = self.act(stack_sub)
        stack_dis = stack_sub.norm(p=self._norm, dim=1)
        stack_sim = - stack_dis
        order_sims = stack_sim.reshape(n_vis, n_label)
        return order_sims


