import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.functional import normalize

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

device = "cuda:6"

torch.set_num_threads(12)


atoms = ["N", "CA", "C", "cent"]

# Last value is the number of atoms in the next residue
angles = [
	("N", "CA", "C"   , 0), ("CA", "C" , "N"   , 1), ("C", "N", "CA", 2),
	("N", "CA", "cent", 0), ("C" , "CA", "cent", 0),
]

# Last value is the number of atoms in the next residue
dihedrals = [
	("C", "N", "CA", "C"   , 3), ("N"   , "CA", "C", "N", 1), ("CA", "C", "N", "CA", 2),
	("C", "N", "CA", "cent", 3), ("cent", "CA", "C", "N", 1),
]

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

class DistanceForces(nn.Module):
	"""
	Calculates forces between two atoms based on their 
		1. atoms types
		2. Euclidian distance
		3. Seperation along the sequence

	Input dim = 50 (24*2 + 2)
	Output dim = 1 (a scalar force)
	"""
	def __init__(self, input_size, hidden_size, output_size):
		super(DistanceForces, self).__init__()


		self.atom_embedding = nn.Sequential(nn.Linear(24,hidden_size),
											nn.LeakyReLU(negative_slope=0.2),
											nn.Linear(hidden_size, hidden_size))

		self.model = nn.Sequential(
			nn.Linear((2*24)+2, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, output_size))

	def forward(self, atom1, atom2, edges):

		#atom1, atom2 = self.atom_embedding(atom1), self.atom_embedding(atom2)

		messages = torch.cat([atom1, atom2, edges], dim=1)

		return self.model(messages)

class AngleForces(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(AngleForces, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, output_size))

	def forward(self, central_atom, angles):

		messages = torch.cat([central_atom, angles[:,:,None]], dim=2)

		return self.model(messages)

class Simulator(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Simulator, self).__init__()

		self.distance_forces = DistanceForces(50, 128, 1)
		self.angle_forces = AngleForces(24+1, 128, 1)

	def forward(self, coords, node_f, res_numbers, masses, seq,
				radius, n_steps, timestep, temperature, animation, device):

		n_atoms = coords.shape[0]
		n_res = n_atoms // len(atoms)
		model_n = 0

		vels = torch.randn(coords.shape).to(device) * temperature
		accs_last = torch.zeros(coords.shape).to(device)
		randn_coords = coords + vels * timestep * n_steps
		loss, passed = rmsd(randn_coords, coords)		

		for i in range(n_steps):

			coords = coords + vels * timestep + 0.5 * accs_last * timestep * timestep

			k = 15
			idx = knn(coords, k+1)
			senders = idx[:,0].repeat_interleave(k)
			receivers = idx[:,1:].reshape(n_atoms*k)

			# Calc Euclidian distance
			diffs = coords[senders] - coords[receivers]
			dists = diffs.norm(dim=1)
			norm_diffs = diffs / dists.clamp(min=0.01).unsqueeze(1)

			# Calc sequence seperation
			seq_sep = abs(res_numbers[senders] - res_numbers[receivers])/5
			mask = seq_sep > 1
			seq_sep[mask] = 1

			# Concat edge features
			edges = torch.cat([dists.unsqueeze(1), seq_sep], dim=1)

			# Compute forces using MLP
			forces = self.distance_forces(node_f[senders], node_f[receivers], edges)
			forces = forces * norm_diffs
			total_forces = forces.view(n_atoms, k, 3).sum(1)/100
			
			batch_size = 1
			atom_types = node_f.view(batch_size, n_res, len(atoms), 24)
			atom_coords = coords.view(batch_size, n_res, 3 * len(atoms))
			atom_accs = torch.zeros(batch_size, n_res, 3 * len(atoms), device=device)
			# Angle forces
			# across_res is the number of atoms in the next residue, starting from atom_3
			for ai, (atom_1, atom_2, atom_3, across_res) in enumerate(angles):
				# Calc vectors and angle between atoms
				ai_1, ai_2, ai_3 = atoms.index(atom_1), atoms.index(atom_2), atoms.index(atom_3)
				if across_res == 0:
					ba = atom_coords[:, :  , (ai_1 * 3):(ai_1 * 3 + 3)] - atom_coords[:, :  , (ai_2 * 3):(ai_2 * 3 + 3)]
					bc = atom_coords[:, :  , (ai_3 * 3):(ai_3 * 3 + 3)] - atom_coords[:, :  , (ai_2 * 3):(ai_2 * 3 + 3)]
				elif across_res == 1:
					ba = atom_coords[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] - atom_coords[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)]
					bc = atom_coords[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] - atom_coords[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)]
				elif across_res == 2:
					ba = atom_coords[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] - atom_coords[:, 1: , (ai_2 * 3):(ai_2 * 3 + 3)]
					bc = atom_coords[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] - atom_coords[:, 1: , (ai_2 * 3):(ai_2 * 3 + 3)]
				ba_norms = ba.norm(dim=2)
				bc_norms = bc.norm(dim=2)
				angs = torch.acos((ba * bc).sum(dim=2) / (ba_norms * bc_norms))
				# Get central atom properties
				if ai == 0 or ai == 3 or ai == 4:
					central_atom_types = atom_types[:,:,1,:]
				elif ai == 1:
					central_atom_types = atom_types[:,:-1,2,:]
				elif ai == 2:
					central_atom_types = atom_types[:,1:,0,:]

				angle_forces = self.angle_forces(central_atom_types, angs)

				cross_ba_bc = torch.cross(ba, bc, dim=2)
				fa = angle_forces * normalize(torch.cross( ba, cross_ba_bc, dim=2), dim=2) / ba_norms.unsqueeze(2)
				fc = angle_forces * normalize(torch.cross(-bc, cross_ba_bc, dim=2), dim=2) / bc_norms.unsqueeze(2)
				fb = -fa -fc
				if across_res == 0:
					atom_accs[:, :  , (ai_1 * 3):(ai_1 * 3 + 3)] += fa
					atom_accs[:, :  , (ai_2 * 3):(ai_2 * 3 + 3)] += fb
					atom_accs[:, :  , (ai_3 * 3):(ai_3 * 3 + 3)] += fc
				elif across_res == 1:
					atom_accs[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] += fa
					atom_accs[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)] += fb
					atom_accs[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] += fc
				elif across_res == 2:
					atom_accs[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] += fa
					atom_accs[:, 1: , (ai_2 * 3):(ai_2 * 3 + 3)] += fb
					atom_accs[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] += fc

			# Calc distance accs
			accs = total_forces/masses.unsqueeze(1)
			# Calc angle accs
			accs += atom_accs.view(n_atoms, 3) / (masses.unsqueeze(1)*100)


			vels = vels + 0.5 * (accs_last + accs) * timestep
			accs_last = accs

			if animation:
				model_n += 1
				save_structure(coords[None,:,:], "animation.pdb", seq, model_n)

		return coords, loss

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
		for protein in tqdm(train_inds):

			coords, node_f, res_numbers, masses, seq = train_set[protein]

			model.train()
			out, basic_loss = model(coords, node_f, res_numbers, masses, seq, 10, 
							n_steps=500, timestep=0.02, temperature=0.02,
							animation=False, device=device)

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

		model.eval()
		with torch.no_grad():
			coords, node_f, res_numbers, masses, seq = get_features("protein_data/example/1CRN.txt", device=device)

			out, basic_loss = model(coords, node_f, res_numbers, masses, seq, 10, 
							n_steps=500, timestep=0.02, temperature=0.2,
							animation=False, device=device)
		

		torch.save(model.state_dict(), os.path.join(model_dir, f"models/model_ang{i}.pt"))

	
		plt.plot(losses)
		plt.xlim(0)
		plt.ylabel("Loss - RMSD (A)")
		plt.xlabel("Epoch")
		plt.title(f'No. epochs = {i+1}')
		plt.legend()
		plt.savefig('with_angles.png')
