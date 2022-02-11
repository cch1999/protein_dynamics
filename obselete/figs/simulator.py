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
    coords, node_f, res_numbers, masses, seq = get_features(
        "protein_data/example/1CRN.txt", device=device
    )

    out, basic_loss = model(
        coords,
        node_f,
        res_numbers,
        masses,
        seq,
        10,
        n_steps=5000,
        timestep=0.02,
        temperature=0.01,
        animation=10,
        device=device,
    )
