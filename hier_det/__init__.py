# import torch
#
# d_vec = 10
#
# # c=[3x10] --- label vec
# c = torch.Tensor([[i for i in range(1,11)],
#                   [i for i in range(11,21)],
#                   [i for i in range(21,31)]])
#
# # b[2x10] --- vis vec
# b = torch.Tensor([[i for i in range(1,11)], [i for i in range(11,21)]])
#
#
# cc = c.repeat(b.size(0), 1)
# bb = b.repeat(1, c.size(0)).reshape(b.size(0) * c.size(0), d_vec)
#
# sub = cc - bb
# dis = sub.norm(p=2, dim=1)
# sim = -dis
#
# sim_mat = sim.reshape(b.size(0), c.size(0))
#
#
#
# print(sim_mat)