import os
import torch
from torch import nn
from open_relation.model.object.model import HypernymVisual


class PredicateVisual(nn.Module):
    def __init__(self,  obj_visual_d, obj_hidden_d, obj_embedding_d, obj_label_vec_path, obj_weights_path,
                        pre_visual_d, pre_hidden_d, pre_embedding_d, pre_label_vec_path):
        super(PredicateVisual, self).__init__()

        # implicit
        self.obj_visual_d = obj_visual_d
        self.obj_embedding = HypernymVisual(obj_visual_d,
                                            obj_hidden_d,
                                            obj_embedding_d,
                                            obj_label_vec_path)

        # load obj embedding weights
        if os.path.isfile(obj_weights_path):
            obj_weights = torch.load(obj_weights_path)
            self.obj_embedding.load_state_dict(obj_weights)
            print('Loading object embedding weights success.')
        else:
            print('No object embedding weights !!!')
            exit(-1)

        # freeze obj embedding
        self.obj_embedding.eval()
        obj_params = self.obj_embedding.parameters()
        for p in obj_params:
            p.requires_grad = False

        # implicit
        # predicate embedding level 1
        self.pre_in_embedding = HypernymVisual(pre_visual_d,
                                               pre_hidden_d,
                                               pre_embedding_d,
                                               pre_label_vec_path)
        # explicit
        # predicate embedding level 2
        self.pre_ex_embedding = HypernymVisual(obj_embedding_d * 2 + pre_embedding_d,
                                               obj_embedding_d * 2 + pre_embedding_d,
                                               pre_embedding_d,
                                               pre_label_vec_path)

    def forward(self, vfs):
        sbj_vfs = vfs[:, :self.obj_visual_d]
        pre_vfs = vfs
        obj_vfs = vfs[:, -self.obj_visual_d:]

        sbj_embedding = self.obj_embedding.embedding(sbj_vfs)
        obj_embedding = self.obj_embedding.embedding(obj_vfs)
        pre_embedding0 = self.pre_in_embedding.embedding(pre_vfs)

        pre_feat = torch.cat([sbj_embedding, pre_embedding0, obj_embedding], 1)
        all_scores, pre_embedding = self.pre_ex_embedding(pre_feat)
        return all_scores, pre_embedding

    def forward2(self, vfs):
        sbj_vfs = vfs[:, :self.obj_visual_d]
        pre_vfs = vfs
        obj_vfs = vfs[:, -self.obj_visual_d:]

        sbj_scores, sbj_embedding = self.obj_embedding.forward(sbj_vfs)
        obj_scores, obj_embedding = self.obj_embedding.forward(obj_vfs)
        pre_embedding0 = self.pre_in_embedding.embedding(pre_vfs)

        pre_feat = torch.cat([sbj_embedding, pre_embedding0, obj_embedding], 1)
        pre_scores, pre_embedding = self.pre_ex_embedding(pre_feat)

        return pre_scores, sbj_scores, obj_scores



