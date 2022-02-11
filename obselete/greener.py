import torch
import matplotlib.pyplot as plt

from cgdms import (
    interactions,
    dist_bin_centres,
    angle_bin_centres,
    dih_bin_centres,
    aas,
)

data = torch.load("cgdms_params_ep45.pt", map_location="cpu")
dists = data["distances"]
angles = data["angles"]
dihedrals = data["dihedrals"]


def get_greener_potential(interaction):
    index = interactions.index(interaction)
    return dist_bin_centres[index], dists[index][1:-1]


def get_greener_angle(angle, aa):
    index = aas.index(aa)
    return angle_bin_centres, angles[angle, index][1:-1]


def get_greener_dihedrals(angle, aa):
    index = aas.index(aa)
    return (
        dih_bin_centres,
        dihedrals[angle, index][1:-1],
    )


interaction = "I_N_L_cent_other"
x, y = get_greener_potential(interaction)
x, y = get_greener_dihedrals(0, "V")

print(0, angle_bin_centres)
print(y)
plt.plot(x, y * 1000)
plt.title(interaction)
plt.savefig("potentials.png")
