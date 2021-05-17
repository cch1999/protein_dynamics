import torch
import numpy as np
from model7 import Simulator, get_features
import matplotlib.pyplot as plt

device = "cuda:6"

model = Simulator(50, 128, 1).to(device)
model.load_state_dict(torch.load("models/current_pot.pt", map_location=device))
model.eval()

with torch.no_grad():

    model = model.angle_forces



    fig, axes = plt.subplots(3,2, figsize=(10, 12))

    atoms = [0,1,2]
    angles = [2.12 ,1.94, 2.02]

    for j, angle in zip(atoms, angles):

        atom = torch.zeros(24).to(device).double()
        atom[[j,4]] = 1
        print(atom)
        forces = []
        angles = np.arange(0, 6.2, 0.01)
            
        for i in angles:
            force = model(atom[None,None,:].float(), torch.tensor(i)[None, None].to(device).float())
            print(force)
            forces.append(force.item())


        

        axes[j,0].plot(angles, forces)
        axes[j,0].axhline(0, color='black')
        axes[j,0].axvline(angle, color='red')
        axes[j,0].set_ylabel("Force")
        axes[j,0].set_xlabel("Bond angle (radians)")
        axes[j,0].text(2.1, 20, 'True angle', color='red')
        axes[j,0].set_title('Force applied to atoms in -C- angle')
        
        pots = []
        y = 0

        for i in range(len(forces)):
            y += -forces[i]
            pots.append(y)
        axes[j,1].plot(angles,pots)
        axes[j,1].axvline(angle, color='red')
        axes[j,1].set_ylabel("Potential energy")
        axes[j,1].set_xlabel("C Bond angle (radians)")
        axes[j,1].text(2.1, 20, 'True angle', color='red')
        axes[j,1].set_title('Potential in -C- angle')
    plt.savefig('c_ang.png')
    