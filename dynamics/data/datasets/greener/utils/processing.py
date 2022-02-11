# import networkx as nx
import os
import torch

import numpy as np
import torch.nn as nn
from pykeops.torch import LazyTensor

import matplotlib.pyplot as plt

from utils.variables import *


def plot_loss(losses, epoch):
    plt.plot(losses)
    plt.ylim(-0.4, 0.01)
    plt.ylabel("Loss - RMSD (A)")
    plt.xlabel("Samples")
    plt.title(f"No. epochs = {epoch}")
    plt.savefig("current_loss.png")


# Read an input data file
# The protein sequence is read from the file but will overrule the file if provided
def read_input_file(fp, seq="", device="cpu"):
    """
    seq: raw sequence
    seq_info: atoms (+ nums) in the sequence by residue number [(0, 'N'), (0, 'CA'), (0, 'C'), (0, 'cent')]
    native_coords: coords
    """

    with open(fp) as f:
        lines = f.readlines()
        if seq == "":
            seq = lines[0].rstrip()
        ss_pred = lines[1].rstrip()
        assert len(seq) == len(
            ss_pred
        ), f"Sequence length is {len(seq)} but SS prediction length is {len(ss_pred)}"

    seq_info = []
    for i in range(len(seq)):
        for atom in atoms:
            seq_info.append((i, atom))
    n_atoms = len(seq_info)
    native_coords = torch.tensor(
        np.loadtxt(fp, skiprows=2), dtype=torch.float, device=device
    ).view(n_atoms, 3)

    masses = []
    for i, r in enumerate(seq):
        mass_CA = 13.0  # Includes H
        mass_N = 15.0  # Includes amide H
        if i == 0:
            mass_N += 2.0  # Add charged N-terminus
        mass_C = 28.0  # Includes carbonyl O
        if i == len(seq) - 1:
            mass_C += 16.0  # Add charged C-terminus
        mass_cent = aa_masses[r] - 74.0  # Subtract non-centroid section
        if r == "G":
            mass_cent += 10.0  # Make glycine artificially heavier
        masses.append(mass_N)
        masses.append(mass_CA)
        masses.append(mass_C)
        masses.append(mass_cent)

    # TODO make this LazyTensor
    masses = torch.tensor(masses, device=device)

    # Different angle potentials for each residue
    inters_ang = torch.tensor(
        [aas.index(r) for r in seq], dtype=torch.long, device=device
    )

    # Different dihedral potentials for each residue and predicted secondary structure type
    inters_dih = torch.tensor(
        [
            aas.index(r) * len(ss_types) + ss_types.index(s)
            for r, s in zip(seq, ss_pred)
        ],
        dtype=torch.long,
        device=device,
    )

    return native_coords, inters_ang, inters_dih, masses, seq


def get_features(fp, device):

    # TODO remove inters constructions
    native_coords, inters_ang, inters_dih, masses, seq = read_input_file(fp)

    one_hot_atoms = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    one_hot_atoms = one_hot_atoms.repeat(len(seq), 1)

    one_hot_seq = torch.zeros(len(seq) * 4, 20)
    for i, aa in enumerate(seq):
        index = aas.index(aa)
        one_hot_seq[i * 4 : (i + 1) * 4, index] = 1

    res_numbers = torch.cat([torch.ones(4, 1) * i for i in range(len(seq))])

    node_f = torch.cat([one_hot_atoms, one_hot_seq], dim=1)

    return (
        native_coords.to(device),
        node_f.to(device),
        res_numbers.to(device),
        masses.to(device),
        seq,
    )


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


def rmsd(c1, c2):
    device = c1.device
    r1 = c1.transpose(0, 1)
    r2 = c2.transpose(0, 1)
    P = r1 - r1.mean(1).view(3, 1)
    Q = r2 - r2.mean(1).view(3, 1)
    cov = torch.matmul(P, Q.transpose(0, 1))
    try:
        U, S, V = torch.svd(cov)
    except RuntimeError:
        report("  SVD failed to converge", 0)
        return torch.tensor([20.0], device=device), False
    d = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, torch.det(torch.matmul(V, U.transpose(0, 1)))],
        ],
        device=device,
    )
    rot = torch.matmul(torch.matmul(V, d), U.transpose(0, 1))
    rot_P = torch.matmul(rot, P)
    diffs = rot_P - Q
    msd = (diffs**2).sum() / diffs.size(1)
    return msd.sqrt(), True


def save_structure(coords, sim_filepath, seq, model_n):
    with open(sim_filepath, "a") as of:
        of.write("MODEL {:>8}\n".format(model_n))
        for ri, r in enumerate(seq):
            for ai, atom in enumerate(atoms):
                of.write(
                    "ATOM   {:>4}  {:<2}  {:3} A{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}  1.00  0.00          {:>2}  \n".format(
                        len(atoms) * ri + ai + 1,
                        atom[:2].upper(),
                        one_to_three_aas[r],
                        ri + 1,
                        coords[0, len(atoms) * ri + ai, 0].item(),
                        coords[0, len(atoms) * ri + ai, 1].item(),
                        coords[0, len(atoms) * ri + ai, 2].item(),
                        atom[0].upper(),
                    )
                )
        of.write("ENDMDL\n")
