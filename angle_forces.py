import torch
import numpy as np
from model7 import Simulator, get_features
import matplotlib.pyplot as plt
import math

device = "cuda:6"

model = Simulator(50, 128, 1).to(device)
model.load_state_dict(torch.load("models/current_pot.pt", map_location=device))
model.eval()

with torch.no_grad():

    model = model.angle_forces



    fig, axes = plt.subplots(1,3, figsize=(13, 4))

    atoms = [[0,1,2],
            [1,2,0],
            [2,0,1]]
    atoms_title = ["Ca", "C", "N"]
    angles = [111 ,116, 122]

    for j, (atom_type, angle) in enumerate(zip(atoms, angles)):

        atom = torch.zeros(24).to(device).double()
        atom[4] = 1
        atom1 = atom.clone()
        atom2 = atom.clone()
        atom3 = atom.clone()
        print(atom_type[0])
        atom1[atom_type[0]] = 1
        atom2[atom_type[1]] = 1
        atom3[atom_type[2]] = 1

        forces = []
        angles = np.arange(-180, 180, 1)
            
        for i in angles:
            i = math.radians(i)
            force = model(atom1[None,None,:].float(), atom2[None,None,:].float(), atom3[None,None,:].float(), torch.tensor(i)[None, None].to(device).float())
            forces.append(force.item())

        axes[j].plot(angles, forces)
        axes[j].set_xlim(-180, 180)
        axes[j].axhline(0, color='black')
        axes[j].axvline(angle, color='red')
        axes[j].set_ylabel("Energy")
        axes[j].set_xlabel("Bond angle (radians)")
        axes[j].text(2.1, 20, 'True angle', color='red')
        axes[j].set_title(f'Learned potential in -{atoms_title[j]}- angle')

    fig.tight_layout()
    plt.savefig('angles.png')
    