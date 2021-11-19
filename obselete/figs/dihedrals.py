import torch
import numpy as np
from model7 import Simulator, get_features
import matplotlib.pyplot as plt
import seaborn as sns

device = "cuda:6"

model = Simulator(50, 128, 1).to(device)
model.load_state_dict(torch.load("models/current_pot.pt", map_location=device))
model.eval()

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)

with torch.no_grad():
    
    # Dihedrals
    model = model.dihedral_forces
    print(model)

    angles = torch.tensor(np.arange(-3.1, 3.1, 0.01), device=device)

    atom = torch.zeros(24).to(device)
    atom1 = atom.clone()
    atom2 = atom.clone()
    atom3 = atom.clone()
    atom4 = atom.clone()

    fig, axes = plt.subplots(1,2, figsize=(12, 4))


    for i in range(2):
        if i == 0:
            atom1[0] = 1
            atom1[23] = 1

            atom2[1] = 1
            atom2[23] = 1

            atom3[2] = 1
            atom3[23] = 1

            atom4[0] = 1
            atom4[23] = 1

        if i == 1:
            atom1[2] = 1
            atom1[23] = 1

            atom2[0] = 1
            atom2[23] = 1

            atom3[1] = 1
            atom3[23] = 1

            atom4[2] = 1
            atom4[23] = 1

        forces = []

        for dist in angles:


            force = model(atom1[None,None,:], atom2[None,None,:],
                            atom3[None,None,:], atom4[None,None,:],
                            torch.tensor([dist])[None,:].to(device).float())


            forces.append(force.item())

        if i == 0:
            phi_pots = forces
            axes[i].set_title("Learned Phi Potential")
        if i == 1:
            psi_pots = forces
            axes[i].set_title("Learned Psi Potential")


        axes[i].plot(angles.cpu(), forces)
        axes[i].axhline(0, color='black')
        axes[i].set_ylabel("Energy")
        axes[i].set_xlabel("Dihedral Angle (Radians)")

        

    plt.savefig('dihedral_angles.png')

    ram = np.zeros([len(phi_pots), len(psi_pots)])

    for i in range(len(psi_pots)):
        for j in range(len(psi_pots)):
            ram[j,i] = psi_pots[i] + phi_pots[j]

    

    fig.clear(True) 
    plt.subplots(1,1, figsize=(6, 5))
    sns.heatmap(ram, vmin=5000, vmax=8000)
    plt.savefig("torsions.png")

    