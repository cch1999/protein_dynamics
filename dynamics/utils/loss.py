import torch


def rmsd(c1, c2):
    """
    Returns the RMSD loss between two structures
    """
    r1 = c1.transpose(0, 1)
    r2 = c2.transpose(0, 1)
    P = r1 - r1.mean(1).view(3, 1)
    Q = r2 - r2.mean(1).view(3, 1)
    cov = torch.matmul(P, Q.transpose(0, 1))
    try:
        U, S, V = torch.svd(cov)
    except RuntimeError:
        report("  SVD failed to converge", 0)
        return torch.tensor([20.0]), False
    d = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, torch.det(torch.matmul(V, U.transpose(0, 1)))],
        ]
    ).to(c1.device)
    rot = torch.matmul(torch.matmul(V, d), U.transpose(0, 1))
    rot_P = torch.matmul(rot, P)
    diffs = rot_P - Q
    msd = (diffs**2).sum() / diffs.size(1)
    return msd.sqrt(), True


def msd(c1, c2):
    """
    Returns the MSD loss between two structures
    """
    r1 = c1.transpose(0, 1)
    r2 = c2.transpose(0, 1)
    P = r1 - r1.mean(1).view(3, 1)
    Q = r2 - r2.mean(1).view(3, 1)
    cov = torch.matmul(P, Q.transpose(0, 1))
    try:
        U, S, V = torch.svd(cov)
    except RuntimeError:
        print("  SVD failed to converge", 0)
        return torch.tensor([20.0]), False
    d = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, torch.det(torch.matmul(V, U.transpose(0, 1)))],
        ]
    ).to(c1.device)
    rot = torch.matmul(torch.matmul(V, d), U.transpose(0, 1))
    rot_P = torch.matmul(rot, P)
    diffs = rot_P - Q
    msd = (diffs**2).sum() / diffs.size(1)
    return msd, True


# TODO Implement FAPE
