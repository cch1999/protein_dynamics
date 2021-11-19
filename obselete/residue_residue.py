import torch
import numpy as np
from model6 import Simulator, get_features
import matplotlib.pyplot as plt

print(torch.cuda.get_device_name(6))
torch.cuda.empty_cache()
aas = [
	"A", "R", "N", "D", "C", "E", "Q", "G", "H", "I",
	"L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
]
n_aas = len(aas)

one_to_three_aas = {
	"C": "CYS", "D": "ASP", "S": "SER", "Q": "GLN", "K": "LYS",
	"I": "ILE", "P": "PRO", "T": "THR", "F": "PHE", "N": "ASN",
	"G": "GLY", "H": "HIS", "L": "LEU", "R": "ARG", "W": "TRP",
	"A": "ALA", "V": "VAL", "E": "GLU", "Y": "TYR", "M": "MET",
}

centroid_dists = {
	"A": 1.5575, "R": 4.3575, "N": 2.5025, "D": 2.5025, "C": 2.0825,
	"E": 3.3425, "Q": 3.3775, "G": 1.0325, "H": 3.1675, "I": 2.3975,
	"L": 2.6075, "K": 3.8325, "M": 3.1325, "F": 3.4125, "P": 1.9075,
	"S": 1.9425, "T": 1.9425, "W": 3.9025, "Y": 3.7975, "V": 1.9775,
}

device = "cuda:6"

model = Simulator(50, 128, 1).to(device)
model.load_state_dict(torch.load("models/current.pt", map_location=device))
model.eval()

with torch.no_grad():
    coords, node_f, res_numbers, masses, seq = get_features("protein_data/example/1CRN.txt", device=device)

    fig, axs = plt.subplots(20,20 , figsize=(16, 16))
    plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
    fig.suptitle('Forces sidechain-sidechain interactions',fontweight ='bold', fontsize=14)

    for i in range(20):
        axs[-1,i].set_xlabel(aas[i], fontweight ='bold',fontsize=11)
        axs[i,0].set_ylabel(aas[i], fontweight ='bold', fontsize=11, rotation=0)




    # Distances
    model = model.distance_forces

    max_dist = 15

    distances = torch.tensor(np.arange(1,max_dist,0.1)).to(device).float()

    for i in range(20):

        atom = torch.zeros(24).to(device)
        atom1 = atom.clone()
        atom2 = atom.clone()
        atom1[3] = 1
        atom1[i+4] = 1

        for j in range(20):
            atom2[3] = 1
            atom2[j+4] = 1

            forces = []
            
            axs[i,j].tick_params(left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False)

            if j > i:
                axs[i,j].spines['bottom'].set_color('white')
                axs[i,j].spines['top'].set_color('white')
                axs[i,j].spines['right'].set_color('white')
                axs[i,j].spines['left'].set_color('white')
            else:
                for dist in distances:
                    force = model(atom1[None,:], atom2[None,:], torch.tensor([dist, 1])[None,:].to(device).float())
                    forces.append(force.item())

                axs[i,j].plot(distances.cpu(), forces, lw=3)
                axs[i,j].axhline(0, color='black')
                axs[i,j].tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
                #axs[i,j].axvline(centroid_dists[aas[i]], color='orange')
                #axs[i//4,i%4].set_ylim(-5, 5)
                #axs[i,j].set_title(one_to_three_aas[aas[i]])
                #axs[i//4,i%4].set_xlim(1,max_dist)
    fig.tight_layout()
    plt.savefig('residues.png')

