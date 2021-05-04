import torch
import torch.nn as nn
from utils import MLP, read_input_file, _compute_connectivity, rmsd
import matplotlib.pyplot as plt
import os
from pykeops.torch import LazyTensor
from tqdm import tqdm

aas = [
	"A", "R", "N", "D", "C", "E", "Q", "G", "H", "I",
	"L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
]

device = "cuda:6"

class DistanceForces(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(DistanceForces, self).__init__()


		self.atom_embedding = nn.Sequential(nn.Linear(24,hidden_size),
											nn.LeakyReLU(negative_slope=0.2),
											nn.Linear(hidden_size, hidden_size))

		self.model = nn.Sequential(
			nn.Linear((2*128)+2, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.Sigmoid(),
			nn.Linear(hidden_size, output_size))

	def forward(self, atom1, atom2, edges):

		atom1, atom2 = self.atom_embedding(atom1), self.atom_embedding(atom2)

		messages = torch.cat([atom1, atom2, edges], dim=1)

		return self.model(messages)


class Simulator(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Simulator, self).__init__()

		self.distance_forces = DistanceForces(50, 128, 1)

	def forward(self, coords, node_f, res_numbers, masses,
				radius, n_steps, timestep, temperature, device):

		n_atoms = coords.shape[0]

		vels = torch.randn(coords.shape).to(device) * temperature
		accs_last = torch.zeros(coords.shape).to(device)
		randn_coords = coords + vels * timestep * n_steps
		loss, passed = rmsd(randn_coords, coords)
		print("Basic loss:", loss)

		for i in range(n_steps):

			coords = coords + vels * timestep + 0.5 * accs_last * timestep * timestep

			k = 20
			idx = knn(coords, k+1)


			senders = idx[:,0].repeat(k)
			receivers = idx[:,1:].reshape(n_atoms*k)

			# Calc Euclidian distance
			diffs = coords[senders] - coords[receivers]
			dists = diffs.norm(dim=1)/radius
			norm_diffs = diffs / dists.clamp(min=0.01).unsqueeze(1)

			# Calc sequence seperation
			seq_sep = abs(res_numbers[senders] - res_numbers[receivers])/5
			mask = seq_sep > 1
			seq_sep[mask] = 1

			edges = torch.cat([dists.unsqueeze(1), seq_sep], dim=1)


			forces = self.distance_forces(node_f[senders], node_f[receivers], edges)
			forces = forces * norm_diffs


			total_forces = forces.view(n_atoms, k, 3).sum(1)

			accs = total_forces/masses.unsqueeze(1)
			
			vels = vels + 0.5 * (accs_last + accs) * timestep
			accs_last = accs

		return coords

def knn(coords, k):
	"""
	Finds the k-nearest neibours
	"""
	coords = coords.to(device)

	N, D = coords.shape
	xyz_i = LazyTensor(coords[:, None, :])
	xyz_j = LazyTensor(coords[None, :, :])

	pairwise_distance_ij = ((xyz_i - xyz_j) ** 2).sum(-1)

	idx = pairwise_distance_ij.argKmin(K=k, axis=1)  # (N, K)

	return idx

def get_features(fp, device):
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

	return native_coords.to(device), node_f.to(device), res_numbers.to(device), masses.to(device), seq

coords, node_f, res_numbers, masses, seq = get_features("protein_data/example/1CRN.txt", device=device)

model = Simulator(50, 128, 1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
model.train()
optimizer.zero_grad()

losses = []

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)

for i in tqdm(range(2000)):
	model.train()

	out = model(coords, node_f, res_numbers, masses, 10, 
					n_steps=250, timestep=0.02, temperature=0.1, device=device)

	loss, passed = rmsd(out, coords)
	loss_log = torch.log(1.0 + loss)
	loss_log.backward()
	optimizer.step()
	optimizer.zero_grad()
	losses.append(loss)

	print("---- Loss",loss)

plt.plot(losses)
plt.savefig("loss.png")
plt.show()