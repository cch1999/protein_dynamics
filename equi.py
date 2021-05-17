import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.functional import normalize
import pdb
from equimodel import EGNN_vel

from random import shuffle

from utils import MLP, read_input_file, _compute_connectivity, rmsd, save_structure
import matplotlib.pyplot as plt
import os
from pykeops.torch import LazyTensor
from tqdm import tqdm

model_dir = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(model_dir, "datasets")
train_val_dir = os.path.join(model_dir, "protein_data", "train_val")
trained_model_file = os.path.join(model_dir, "test_model2.pt")

train_proteins = [l.rstrip() for l in open(os.path.join(dataset_dir, "train.txt"))]
val_proteins   = [l.rstrip() for l in open(os.path.join(dataset_dir, "val.txt"  ))]

device = "cuda:5"

atoms = ["N", "CA", "C", "cent"]


aas = [
	"A", "R", "N", "D", "C", "E", "Q", "G", "H", "I",
	"L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
]

n_aas = len(aas)

class ProteinDataset(Dataset):
    def __init__(self, pdbids, coord_dir, device="cpu"):
        self.pdbids = pdbids
        self.coord_dir = coord_dir
        self.set_size = len(pdbids)
        self.device = device

    def __len__(self):
        return self.set_size

    def __getitem__(self, index):
        fp = os.path.join(self.coord_dir, self.pdbids[index] + ".txt")
        return get_features(fp, device=self.device)


class Simulator(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Simulator, self).__init__()
		from egnn_pytorch.egnn_pytorch import EGNN_Network

		self.net = EGNN_vel(24, 8, 64)

	def forward(self, coords, feats, res_numbers, masses, seq,
				radius, n_steps, timestep, temperature, animation, device):

		n_atoms = coords.shape[0]
		n_res = n_atoms // len(atoms)
		model_n = 0

		vels = torch.randn(coords.shape).to(device) * temperature
		accs_last = torch.zeros(coords.shape).to(device)
		randn_coords = coords + vels * timestep * n_steps
		loss, passed = rmsd(randn_coords, coords)		

		coords = randn_coords

		for i in range(n_steps):
			print(feats.shape)
			print(feats)
			edges = knn(coords, 15)
			edge_attr = edge_attributes(coords, edges, res_numbers)
			coords, vels = self.net(feats.unsqueeze(0), coords, edges, vels, edge_attr) 

		return coords[0], loss

def edge_attributes(coords, edges, res_numbers):
	senders, receivers = edges[0], edges[1]
	diffs = coords[senders] - coords[receivers]
	dists = diffs.norm(dim=1)
	norm_diffs = diffs / dists.clamp(min=0.01).unsqueeze(1)

	# Calc sequence seperation
	seq_sep = abs(res_numbers[senders] - res_numbers[receivers])/5
	mask = seq_sep > 1
	seq_sep[mask] = 1

	# Concat edge features
	edges = torch.cat([dists, seq_sep], dim=1)

	return edges

def knn(coords, k):
	"""
	Finds the k-nearest neibours
	"""

	N, D = coords.shape
	xyz_i = LazyTensor(coords[:, None, :])
	xyz_j = LazyTensor(coords[None, :, :])

	pairwise_distance_ij = ((xyz_i - xyz_j) ** 2).sum(-1)

	idx = pairwise_distance_ij.argKmin(K=k, axis=1)  # (N, K)

	senders = idx[:,0].repeat_interleave(k-1)
	receivers = idx[:,1:].reshape(N*(k-1))

	return [senders, receivers]

def get_features(fp, device):


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

if __name__ == "__main__":

	saving = 25

	data_dir = "protein_data/train_val/"
	data = os.listdir(data_dir)

	model = Simulator(50, 128, 1).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
	losses = []

	pytorch_total_params = sum(p.numel() for p in model.parameters())
	print(pytorch_total_params)

	train_set = ProteinDataset(train_proteins, train_val_dir, device=device)
	val_set   = ProteinDataset(val_proteins  , train_val_dir, device=device)

	for i in range(20):
		print(f"Starting Epoch {i}:")

		train_inds = list(range(len(train_set)))
		val_inds   = list(range(len(val_set)))
		shuffle(train_inds)
		shuffle(val_inds)
		model.train()
		optimizer.zero_grad()
		for batch, protein in tqdm(enumerate(train_inds)):

			coords, node_f, res_numbers, masses, seq = train_set[protein]

			model.train()
			out, basic_loss = model(coords, node_f, res_numbers, masses, seq, 10, 
							n_steps=1, timestep=0.02, temperature=0.5,
							animation=None, device=device)
			print(out.shape)
			print(coords.shape)
			loss, passed = rmsd(out, coords)
			loss_log = torch.log(1.0 + loss)
			loss_log.backward()
			optimizer.step()
			optimizer.zero_grad()
			losses.append(loss - basic_loss)

			print("Epoch:", i)
			print("Basic loss:", round(basic_loss.item(),3))
			print("----- Loss:", round(loss.item(),3))
			print("-Loss diff:", round(loss.item() - basic_loss.item(), 3))

			if batch % saving == 0:
				torch.save(model.state_dict(), os.path.join(model_dir, f"models/equi.pt"))
				plt.plot(losses)
				plt.ylabel("Loss - RMSD (A)")
				plt.xlabel("Batches")
				plt.ylim(-0.001, 0.001)
				plt.title(f'No. epochs = {i+1}')
				plt.savefig('equi_loss.png')


		model.eval()
		with torch.no_grad():
			coords, node_f, res_numbers, masses, seq = get_features("protein_data/example/1CRN.txt", device=device)

			out, basic_loss = model(coords, node_f, res_numbers, masses, seq, 10, 
							n_steps=500, timestep=0.02, temperature=0.2,
							animation=False, device=device)
		

		torch.save(model.state_dict(), os.path.join(model_dir, f"models/model_dih{i}.pt"))

	
		plt.plot(losses)
		plt.ylabel("Loss - RMSD (A)")
		plt.xlabel("Epoch")
		plt.title(f'No. epochs = {i+1}')
		plt.legend()
		plt.savefig('with_angles.png')
