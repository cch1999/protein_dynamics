import torch
x = torch.randn(184, 3)

from utils import read_input_file

native_coords, inters_ang, inters_dih, masses, seq = read_input_file("protein_data/example/1CRN.txt")

print("AHHHH")
# Turn our Tensors into KeOps symbolic variables:
from pykeops.torch import LazyTensor
import numpy as np
import time

from pykeops.torch.cluster import grid_cluster
from pykeops.torch.cluster import cluster_ranges_centroids
from pykeops.torch.cluster import sort_clusters
from utils import rmsd

x = native_coords
y = native_coords

coords = native_coords

vels = torch.zeros(native_coords.shape)

time_steps = 1000

xyz_i = LazyTensor(coords[:,None,:])  # x_i.shape = (1e6, 1, 3)
xyz_j = LazyTensor(coords[None,:,:])  # y_j.shape = ( 1, 2e6,3)

# We can now perform large-scale computations, without memory overflows:
D_ij = ((xyz_i - xyz_j)**2).sum(-1).sqrt()
V_ij = xyz_i - xyz_j

mask = (4-D_ij).step()
V_ij = V_ij * mask

obj = torch.randn([128,50])

orr = obj.t().mm(mask)

print(orr)

exit()
# Normalise force
F_ij = V_ij.normalize()

# Calc net force and acceleration
F = F_ij.sum(1)
acc = F/masses[:, None]


exit()

for i in range(time_steps):

    xyz_i = LazyTensor(coords[:,None,:])  # x_i.shape = (1e6, 1, 3)
    xyz_j = LazyTensor(coords[None,:,:])  # y_j.shape = ( 1, 2e6,3)

    # We can now perform large-scale computations, without memory overflows:
    D_ij = ((xyz_i - xyz_j)**2).sum(-1).sqrt()
    V_ij = xyz_i - xyz_j

    mask = (4-D_ij).step()
    V_ij = V_ij * mask


    # Normalise force
    F_ij = V_ij.normalize()

    # Calc net force and acceleration
    F = F_ij.sum(1)
    acc = F/masses[:, None]

    # Intergrator (assuming dT = 1)
    vels = vels + acc
    coords = coords + vels

    print(rmsd(native_coords, coords)[0])


"""

eps = 0.05  # Size of our square bins

x_labels = grid_cluster(x, eps)  # class labels
y_labels = grid_cluster(y, eps)  # class labels

# Compute one range and centroid per class:
x_ranges, x_centroids, _ = cluster_ranges_centroids(x, x_labels)
y_ranges, y_centroids, _ = cluster_ranges_centroids(y, y_labels)

x, x_labels = sort_clusters(x, x_labels)
y, y_labels = sort_clusters(y, y_labels)

sigma = 0.05  # Characteristic length of interaction

# Compute a coarse Boolean mask:
D = ((x_centroids[:, None, :] - y_centroids[None, :, :]) ** 2).sum(2)
#keep = D < (4 * sigma) ** 2

keep = D < 10



from pykeops.torch.cluster import from_matrix

ranges_ij = from_matrix(x_ranges, y_ranges, keep)

areas = (x_ranges[:, 1] - x_ranges[:, 0])[:, None] * (y_ranges[:, 1] - y_ranges[:, 0])[None, :]


V = (x_ranges[:, 1] - x_ranges[:, 0])[:, None] - (y_ranges[:, 1] - y_ranges[:, 0])[None, :]




total_area = areas.sum().item()  # should be equal to N*M
sparse_area = areas[keep].sum().item()
"""

