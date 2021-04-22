import torch
x = torch.randn(1000, 3, requires_grad=True)

# Turn our Tensors into KeOps symbolic variables:
from pykeops.torch import LazyTensor

xyz_i = LazyTensor(x[:,None,:] )  # x_i.shape = (1e6, 1, 3)
xyz_j = LazyTensor(x[None,:,:] )  # y_j.shape = ( 1, 2e6,3)

# We can now perform large-scale computations, without memory overflows:
D_ij = ((xyz_i - xyz_j)**2).sum(-1).sqrt()
V_ij = xyz_i - xyz_j

print(D_ij)

D_ij = D_ij < 4.0

print(D_ij)