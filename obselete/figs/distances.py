import torch
import numpy as np
from model7 import Simulator
import sys
import matplotlib.pyplot as plt

device = "cuda:6"

model = Simulator(50, 128, 1).to(device)
model.load_state_dict(torch.load("models/current_pot.pt", map_location=device))
model.eval()

with torch.no_grad():
    # Distances
    model = model.distance_forces
    print(model)

    pairs = [[0, 1], [1, 2], [0, 2]]
    titles = ["N-Ca", "Ca-C", "C-N"]
    true_dists = [1.46, 1.51, 1.32]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i in range(3):

        distances = torch.tensor(np.arange(0, 5, 0.1)).to(device).float()

        atom = torch.zeros(24).to(device)
        atom1 = atom.clone()
        atom2 = atom.clone()
        atom1[pairs[i][0]] = 1
        atom1[4] = 1
        atom2[pairs[i][1]] = 1
        atom2[4] = 1

        forces = []

        for dist in distances:

            force = model(
                atom1[None, :],
                atom2[None, :],
                torch.tensor([dist, 0])[None, :].to(device).float(),
            )
            forces.append(force.item())

        axes[i].plot(distances.cpu(), forces, label="Learned potential")
        # axes[i].axhline(0, color='black')
        axes[i].axvline(true_dists[i], color="red", label="True distance", ls="--")
        axes[i].set_ylabel("Energy")
        axes[i].set_xlabel("Bond distance (Å)")
        # axes[i].text(1.7, -100, 'True distsance', color='red')
        axes[i].set_title(f"Potential between Alanine {titles[i]} atoms")
        axes[i].legend()

    fig.tight_layout()
    plt.savefig("dists.png")
