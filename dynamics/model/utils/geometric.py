"""
Geometric functions implemented in Keops
"""
from pykeops.torch import LazyTensor


def knn(coords, k):
    """
    Finds the k-nearest neibours
    """

    N, D = coords.shape
    xyz_i = LazyTensor(coords[:, None, :])
    xyz_j = LazyTensor(coords[None, :, :])

    pairwise_distance_ij = ((xyz_i - xyz_j) ** 2).sum(-1)

    idx = pairwise_distance_ij.argKmin(K=k, axis=1)  # (N, K)

    return idx
