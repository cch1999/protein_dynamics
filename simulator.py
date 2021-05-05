import torch
from model2 import Simulator, get_features

device = "cuda:5"

model = Simulator(50, 128, 1).to(device)
model.load_state_dict(torch.load("model0.pt"))
model.eval()

with torch.no_grad():
    coords, node_f, res_numbers, masses, seq = get_features("protein_data/example/1CRN.txt", device=device)

    out, basic_loss = model(coords, node_f, res_numbers, masses, seq, 10, 
                    n_steps=500, timestep=0.02, temperature=0.2,
                    animation=True, device=device)

    print(out)