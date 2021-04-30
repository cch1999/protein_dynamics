import torch
import torch.nn as nn
from utils import MLP, read_input_file, _compute_connectivity, rmsd
import matplotlib.pyplot as plt
import os

aas = [
	"A", "R", "N", "D", "C", "E", "Q", "G", "H", "I",
	"L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
]


class DistanceForces(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(DistanceForces, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, output_size))

	def forward(self, messages):

		return self.model(messages)


class Simulator(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Simulator, self).__init__()

		self.distance_forces = DistanceForces(50, 128, 1)

	def forward(self, coords, node_f, res_numbers, masses,
				radius, n_steps, timestep, temperature):

		vels = torch.randn(coords.shape) * temperature
		accs_last = torch.zeros(coords.shape)

		for i in range(n_steps):

			coords = coords + vels * timestep + 0.5 * accs_last * timestep * timestep

			senders, receivers = _compute_connectivity(coords.detach().numpy(), 10, False)

			# Calc Euclidian distance
			diffs = coords[senders] - coords[receivers]
			dists = diffs.norm(dim=1)/radius
			norm_diffs = diffs / dists.clamp(min=0.01).unsqueeze(1)

			# Calc sequence seperation
			seq_sep = abs(res_numbers[senders] - res_numbers[receivers])/5
			mask = seq_sep > 1
			seq_sep[mask] = 1

			messages = torch.cat([node_f[senders], node_f[receivers], dists.unsqueeze(1), seq_sep], dim=1)

			forces = self.distance_forces(messages)
			forces = forces * norm_diffs
			total_forces = torch.zeros(coords.shape)
			total_forces[receivers] += forces
			accs = total_forces/masses.unsqueeze(1)
			
			vels = vels + 0.5 * (accs_last + accs) * timestep
			accs_last = accs

		return coords



def get_features(fp):
	"""
	Returns a PyTorch Geometric Data object encoding a course-grained protein graph.
	"""

	native_coords, inters_ang, inters_dih, masses, seq = read_input_file(fp)

	one_hot_atoms = torch.tensor([[1,0,0,0],
								[0,1,0,0],
								[0,0,1,0],
								[0,0,0,1]])
	one_hot_atoms = one_hot_atoms.repeat(len(seq), 1)

	one_hot_seq = torch.zeros(len(seq)*4, 20)
	for i, aa in enumerate(seq):
		index = aas.index(aa)
		one_hot_seq[i*4:(i+1)*4, index] = 1

	res_numbers = torch.cat([torch.ones(4,1)*i for i in range(len(seq))])

	node_f = torch.cat([one_hot_atoms, one_hot_seq], dim=1)

	return native_coords, node_f, res_numbers, masses, seq

coords, node_f, res_numbers, masses, seq = get_features("protein_data/example/1CRN.txt")

model = Simulator(50, 256, 1)

optimizer = torch.optim.Adam(model.parameters())
model.train()
optimizer.zero_grad()

losses = []

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)

for i in range(100):

	out = model(coords, node_f, res_numbers, masses, 10, 40, 0.002, 0.1)

	loss, passed = rmsd(out, coords)
	loss_log = torch.log(1.0 + loss)
	loss_log.backward(retain_graph=True)
	optimizer.step()
	optimizer.zero_grad()
	losses.append(loss)

	print(loss)

plt.plot(losses)
plt.show()