import torch
import numpy as np
from model6 import Simulator, get_features
import matplotlib.pyplot as plt
from cgdms import starting_coords

device = "cuda:6"

model = Simulator(50, 128, 1).to(device)
model.load_state_dict(torch.load("models/current.pt", map_location=device))
model.eval()

with torch.no_grad():
    coords, node_f, res_numbers, masses, seq = get_features("protein_data/example/1CRN.txt", device=device)

    coords = starting_coords(seq, conformation="extended", device=device)
    
    out, basic_loss = model(coords, node_f, res_numbers, masses, seq, 10, 
                    n_steps=100000, timestep=0.02, temperature=0.02,
                    animation=200, device=device)
    