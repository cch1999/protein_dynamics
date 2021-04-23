from utils import read_input_file
import torch
import numpy as np

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
    d = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, torch.det(torch.matmul(V, U.transpose(0, 1)))]
    ], device=device)
    rot = torch.matmul(torch.matmul(V, d), U.transpose(0, 1))
    rot_P = torch.matmul(rot, P)
    diffs = rot_P - Q
    msd = (diffs ** 2).sum() / diffs.size(1)
    return msd.sqrt(), True

time_steps = 10
temperature = 1

native_coords, inters_ang, inters_dih, masses, seq = read_input_file("protein_data/example/1CRN.txt")

coords = native_coords
vels = torch.zeros(coords.shape)


for i in range(time_steps):

    D_ij = ((coords[:,None,:]-coords[None,:,:])**2).sum(-1).sqrt()
    V_ij = coords[:,None,:]-coords[None,:,:]

    mask = D_ij < 4.0
    F_ij = V_ij

    mag = (F_ij**2).sum(-1).sqrt()
    
    # Normalise force
    F_ij = (F_ij/mag[:,:,None])  * mask[:,:,None]

    # Calc acceleration
    forces = (F_ij**2).nansum(1)
    acc = forces/masses[:, None]

    # Update coords
    vels = vels + acc
    coords = coords + vels

    print(rmsd(coords, native_coords)[0])