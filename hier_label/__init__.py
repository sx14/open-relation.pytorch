# import torch
# import numpy as np
# from torch import nn
#
# a = nn.CrossEntropyLoss()
#
#
# v = [[-10, -10, -10, -10, -10, -10]]
# y = [0]
#
# v1 = torch.from_numpy(np.array(v))
# vv1 = torch.autograd.Variable(v1).float()
# y1 = torch.from_numpy(np.array(y))
# yy1 = torch.autograd.Variable(y1).long()
#
# l = a.forward(vv1, yy1)
# print(l.data.numpy().tolist())

# from nltk.corpus import wordnet as wn
#
# a = wn.synset('person.n.01')
# print(a.hypernym_paths())