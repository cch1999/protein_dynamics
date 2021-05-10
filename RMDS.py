import torch
import numpy as np
from model6 import Simulator, get_features
import matplotlib.pyplot as plt
from utils import rmsd

device = "cuda:6"

model = Simulator(50, 128, 1).to(device)
model.load_state_dict(torch.load("models/current.pt", map_location=device))
model.eval()


with torch.no_grad():
    coords, node_f, res_numbers, masses, seq = get_features("protein_data/example/1CRN.txt", device=device)

    native_coords = coords
    losses = []
    
    for i in range(0, 10000, 500):
        print(i)
        out, basic_loss = model(coords, node_f, res_numbers, masses, seq, 10, 
                        n_steps=i, timestep=0.02, temperature=0.2,
                        animation=False, device=device)
  
        loss = rmsd(native_coords,out)
        print(loss)
        losses.append(loss)

    plt.plot(losses)
    plt.savefig('RMSDs.png')