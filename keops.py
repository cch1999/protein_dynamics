import torch
x = torch.randn(184, 3)

from utils import read_input_file

coords = read_input_file("protein_data/example/1CRN.txt")[0]


# Turn our Tensors into KeOps symbolic variables:
from pykeops.torch import LazyTensor
import numpy as np
import time

from pykeops.torch.cluster import grid_cluster
from pykeops.torch.cluster import cluster_ranges_centroids
from pykeops.torch.cluster import sort_clusters


x = coords
y = coords

xyz_i = LazyTensor(x[:,None,:])  # x_i.shape = (1e6, 1, 3)
xyz_j = LazyTensor(x[None,:,:])  # y_j.shape = ( 1, 2e6,3)

# We can now perform large-scale computations, without memory overflows:
D_ij = ((xyz_i - xyz_j)**2).sum(-1).sqrt()
V_ij = xyz_i - xyz_j

print(D_ij[:,:,None])

"""
Keops tutorial
"""


exit()


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

print(coords.shape)

print(keep[:10,:10])
print(keep.shape)

from pykeops.torch.cluster import from_matrix

ranges_ij = from_matrix(x_ranges, y_ranges, keep)

areas = (x_ranges[:, 1] - x_ranges[:, 0])[:, None] * (y_ranges[:, 1] - y_ranges[:, 0])[None, :]

print(ranges_ij)

V = (x_ranges[:, 1] - x_ranges[:, 0])[:, None] - (y_ranges[:, 1] - y_ranges[:, 0])[None, :]

print(V.shape)
print(V_ij[0])
exit()

print((x_ranges[:, 1] - x_ranges[:, 0])[None, :])
print(areas)

"""
total_area = areas.sum().item()  # should be equal to N*M
sparse_area = areas[keep].sum().item()
"""

