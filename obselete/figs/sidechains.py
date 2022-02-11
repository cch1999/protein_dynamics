import torch
import numpy as np
from model7 import Simulator, get_features
import matplotlib.pyplot as plt

print(torch.cuda.get_device_name(6))
torch.cuda.empty_cache()
aas = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
n_aas = len(aas)

one_to_three_aas = {
    "C": "CYS",
    "D": "ASP",
    "S": "SER",
    "Q": "GLN",
    "K": "LYS",
    "I": "ILE",
    "P": "PRO",
    "T": "THR",
    "F": "PHE",
    "N": "ASN",
    "G": "GLY",
    "H": "HIS",
    "L": "LEU",
    "R": "ARG",
    "W": "TRP",
    "A": "ALA",
    "V": "VAL",
    "E": "GLU",
    "Y": "TYR",
    "M": "MET",
}

centroid_dists = {
    "A": 1.5575,
    "R": 4.3575,
    "N": 2.5025,
    "D": 2.5025,
    "C": 2.0825,
    "E": 3.3425,
    "Q": 3.3775,
    "G": 1.0325,
    "H": 3.1675,
    "I": 2.3975,
    "L": 2.6075,
    "K": 3.8325,
    "M": 3.1325,
    "F": 3.4125,
    "P": 1.9075,
    "S": 1.9425,
    "T": 1.9425,
    "W": 3.9025,
    "Y": 3.7975,
    "V": 1.9775,
}

device = "cuda:6"

model = Simulator(50, 128, 1).to(device)
model.load_state_dict(torch.load("models/current_pot.pt", map_location=device))
model.eval()

with torch.no_grad():
    coords, node_f, res_numbers, masses, seq = get_features(
        "protein_data/example/1CRN.txt", device=device
    )

    fig, axs = plt.subplots(5, 4, figsize=(11, 9))
    fig.suptitle(
        "Learned potentials between Ca atoms and side chains centroids - orange line is optimum distance"
    )

    # Distances
    model = model.distance_forces

    max_dist = 5

    distances = torch.tensor(np.arange(1, max_dist, 0.01)).to(device).float()

    for i in range(20):

        atom = torch.zeros(24).to(device)
        atom1 = atom.clone()
        atom2 = atom.clone()
        atom1[1] = 1
        atom1[i + 4] = 1
        atom2[3] = 1
        atom2[i + 4] = 1

        forces = []

        for dist in distances:

            force = model(
                atom1[None, :],
                atom2[None, :],
                torch.tensor([dist, 0])[None, :].to(device).float(),
            )

            forces.append(force.item())

        axs[i // 4, i % 4].plot(distances.cpu(), forces)
        axs[i // 4, i % 4].axhline(0, color="black")
        axs[i // 4, i % 4].axvline(centroid_dists[aas[i]], color="orange")
        # axs[i//4,i%4].set_ylim(-15, 5)
        axs[i // 4, i % 4].set_title(one_to_three_aas[aas[i]])
        axs[i // 4, i % 4].set_xlim(1, max_dist)

    for ax in axs.flat:
        ax.set(xlabel="Distance (Å)", ylabel="Energy")
    for ax in axs.flat:
        ax.label_outer()
    fig.tight_layout()

    plt.savefig("sidechains.png")
