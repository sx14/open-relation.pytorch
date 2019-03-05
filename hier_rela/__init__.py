import torch

a = torch.Tensor([5])
b = torch.Tensor([5])

c = torch.stack((a,b), dim=1)