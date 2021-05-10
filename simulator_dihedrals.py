import torch
import numpy as np
from model6 import Simulator, get_features
import matplotlib.pyplot as plt
import seaborn as sns

device = "cuda:6"

model = Simulator(50, 128, 1).to(device)
model.load_state_dict(torch.load("models/current.pt", map_location=device))
model.eval()

with torch.no_grad():

    # Dihedrals
    model = model.dihedral_forces
    print(model)

    angles = torch.tensor(np.arange(-3.2, 3.2, 0.01), device=device)

    atom = torch.zeros(24).to(device)
    atom1 = atom.clone()
    atom2 = atom.clone()
    atom3 = atom.clone()
    atom4 = atom.clone()
    atom1[0] = 1
    atom1[5] = 1

    atom2[1] = 1
    atom2[5] = 1

    atom3[2] = 1
    atom3[5] = 1

    atom4[0] = 1
    atom4[5] = 1

    forces = []

    for dist in angles:
        force = model(atom1[None,None,:], atom2[None,None,:],
                        atom3[None,None,:], atom4[None,None,:],
                        torch.tensor([dist])[None,:].to(device).float())

        forces.append(force.item())
    
    pots1 = []
    y = 0
    for i in range(len(forces)):
        y += forces[i]
        pots1.append(y)


    atom = torch.zeros(24).to(device)
    atom1 = atom.clone()
    atom2 = atom.clone()
    atom3 = atom.clone()
    atom4 = atom.clone()
    atom1[2] = 1
    atom1[5] = 1

    atom2[0] = 1
    atom2[5] = 1

    atom3[1] = 1
    atom3[5] = 1

    atom4[2] = 1
    atom4[5] = 1

    forces = []

    for dist in angles:
        force = model(atom1[None,None,:], atom2[None,None,:],
                        atom3[None,None,:], atom4[None,None,:],
                        torch.tensor([dist])[None,:].to(device).float())

        forces.append(force.item())
    
    pots2 = []
    y = 0
    for i in range(len(forces)):
        y += forces[i]
        pots2.append(y)

    print(len(pots1))
    print(len(pots2))
 
    pots = np.zeros([620,620])

    for i in range(620):
        for j in range(620):
            pots[j,i] = pots1[i] + pots2[j]

    print(pots)
    sns.heatmap(pots)
    plt.savefig("torsions.png")