import torch
import numpy as np
from model6 import Simulator
import sys
import matplotlib.pyplot as plt

device = "cuda:6"

model = Simulator(50, 128, 1).to(device)
model.load_state_dict(torch.load("models/current.pt", map_location=device))
model.eval()

with torch.no_grad():
    # Distances
    model = model.distance_forces
    print(model)

    pairs = [[0,1], [1,2], [0,2]]
    fig, axes = plt.subplots(3,2, figsize=(12, 16))


    for i in range(3):

        distances = torch.tensor(np.arange(0,2,0.1)).to(device).float()

        atom = torch.zeros(24).to(device)
        atom1 = atom.clone()
        atom2 = atom.clone()
        atom1[pairs[i][0]] = 1
        atom1[4] = 1
        atom2[pairs[i][1]] = 1
        atom2[4] = 1
        print(atom1)
        print(atom2)

        forces = []

        for dist in distances:

            force = model(atom1[None,:], atom2[None,:], torch.tensor([dist, 0])[None,:].to(device).float())

            print(force)

            forces.append(force.item())


        axes[i,0].plot(distances.cpu(), forces)
        axes[i,0].axhline(0, color='black')
        axes[i,0].axvline(1.51, color='red')
        axes[i,0].set_ylabel("Force")
        axes[i,0].set_ylim(-1,1)
        axes[i,0].set_xlabel("Bond distance (Å)")
        axes[i,0].text(1.7, 5, 'True distsance', color='red')
        axes[i,0].set_title('Force between Alanine R-Ca atoms')
        
        pots = []
        y = 0
        print(forces)
        for force in forces:
            print(i)
            y += -force
            pots.append(y)
        print(pots)
        print(len(pots))
        axes[i,1].plot(distances.cpu(), pots)
        axes[i,1].axvline(1.51, color='red')
        axes[i,1].set_ylabel("Potential energy")
        axes[i,1].set_xlabel("Bond distance (Å)")
        axes[i,1].text(1.7, -20, 'True distance', color='red')
        axes[i,1].set_title('Potential between Alanine R-Ca atoms')
    plt.savefig('dists.png')
 