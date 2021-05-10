import torch
import numpy as np
from model6 import Simulator, get_features
import matplotlib.pyplot as plt

device = "cuda:6"

model = Simulator(50, 128, 1).to(device)
model.load_state_dict(torch.load("models/current.pt", map_location=device))
model.eval()

with torch.no_grad():

    model = model.angle_forces

    atom = torch.zeros(24).to(device).double()
    atom[[1,4]] = 1
    print(atom)
    forces = []
    angles = np.arange(0, 6.2, 0.01)
    for i in angles:
        force = model(atom[None,None,:].float(), torch.tensor(i)[None, None].to(device).float())
        print(force)
        forces.append(force.item())
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 4))

    ax1.plot(angles, forces)
    ax1.axhline(0, color='black')
    ax1.axvline(1.94, color='red')
    ax1.set_ylabel("Force")
    ax1.set_xlabel("C Bond angle (radians)")
    ax1.text(2.1, 20, 'C bond angle', color='red')
    ax1.set_title('Force applied to atoms in -C- angle')
    
    pots = []
    y = 0

    for i in range(len(forces)):
        print(i)
        y += -forces[i]
        pots.append(y)
    ax2.plot(angles,pots)
    ax2.axvline(1.94, color='red')
    ax2.set_ylabel("Potential energy")
    ax2.set_xlabel("C Bond angle (radians)")
    ax2.text(2.1, 20, 'C bond angle', color='red')
    ax2.set_title('Potential in -C- angle')
    plt.savefig('c_ang.png')
    