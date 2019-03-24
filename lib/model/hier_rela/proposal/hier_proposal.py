import torch
from torch import nn


class HierProposal(nn.Module):
    def __init__(self, input_len):
        super(HierProposal, self).__init__()

        output_len = 1

        self.hidden = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(input_len, input_len))

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(input_len, output_len),
            nn.Sigmoid())

    def forward(self, sbj_vec, obj_vec):
        sbj_obj_vec = torch.cat([sbj_vec, obj_vec], 1)
        hidden = self.hidden(sbj_obj_vec)
        scores = self.output(hidden)
        return scores

