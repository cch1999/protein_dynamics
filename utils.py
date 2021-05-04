#import networkx as nx
import numpy as np
import os
import torch
from sklearn import neighbors
import matplotlib.pyplot as plt
import torch.nn as nn
#from torch_geometric.data import Data
from pykeops.torch import LazyTensor



cgdms_dir = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(cgdms_dir, "datasets")
train_val_dir = os.path.join(cgdms_dir, "protein_data", "train_val")

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

one_to_three_aas = {
	"C": "CYS", "D": "ASP", "S": "SER", "Q": "GLN", "K": "LYS",
	"I": "ILE", "P": "PRO", "T": "THR", "F": "PHE", "N": "ASN",
	"G": "GLY", "H": "HIS", "L": "LEU", "R": "ARG", "W": "TRP",
	"A": "ALA", "V": "VAL", "E": "GLU", "Y": "TYR", "M": "MET",
}
three_to_one_aas = {one_to_three_aas[k]: k for k in one_to_three_aas}

aa_masses = {
	"A": 89.09 , "R": 174.2, "N": 132.1, "D": 133.1, "C": 121.2,
	"E": 147.1 , "Q": 146.1, "G": 75.07, "H": 155.2, "I": 131.2,
	"L": 131.2 , "K": 146.2, "M": 149.2, "F": 165.2, "P": 115.1,
	"S": 105.09, "T": 119.1, "W": 204.2, "Y": 181.2, "V": 117.1,
}

ss_types = ["H", "E", "C"]

# Minima in the learned potential after 45 epochs of training
centroid_dists = {
	"A": 1.5575, "R": 4.3575, "N": 2.5025, "D": 2.5025, "C": 2.0825,
	"E": 3.3425, "Q": 3.3775, "G": 1.0325, "H": 3.1675, "I": 2.3975,
	"L": 2.6075, "K": 3.8325, "M": 3.1325, "F": 3.4125, "P": 1.9075,
	"S": 1.9425, "T": 1.9425, "W": 3.9025, "Y": 3.7975, "V": 1.9775,
}



# Read an input data file
# The protein sequence is read from the file but will overrule the file if provided
def read_input_file(fp, seq="", device="cpu"):
	"""
	seq: raw sequence
	seq_info: atoms (+ nums) in the sequence by residue number [(0, 'N'), (0, 'CA'), (0, 'C'), (0, 'cent')]
	native_coords: coords
	"""

	with open(fp) as f:
		lines = f.readlines()
		if seq == "":
			seq = lines[0].rstrip()
		ss_pred = lines[1].rstrip()
		assert len(seq) == len(ss_pred), f"Sequence length is {len(seq)} but SS prediction length is {len(ss_pred)}"
	
	seq_info = []
	for i in range(len(seq)):
		for atom in atoms:
			seq_info.append((i, atom))
	n_atoms = len(seq_info)
	native_coords = torch.tensor(np.loadtxt(fp, skiprows=2), dtype=torch.float,
									device=device).view(n_atoms, 3)


	masses = []
	for i, r in enumerate(seq):
		mass_CA = 13.0 # Includes H
		mass_N = 15.0 # Includes amide H
		if i == 0:
			mass_N += 2.0 # Add charged N-terminus
		mass_C = 28.0 # Includes carbonyl O
		if i == len(seq) - 1:
			mass_C += 16.0 # Add charged C-terminus
		mass_cent = aa_masses[r] - 74.0 # Subtract non-centroid section
		if r == "G":
			mass_cent += 10.0 # Make glycine artificially heavier
		masses.append(mass_N)
		masses.append(mass_CA)
		masses.append(mass_C)
		masses.append(mass_cent)

	# TODO make this LazyTensor
	masses = torch.tensor(masses, device=device)

	# Different angle potentials for each residue
	inters_ang = torch.tensor([aas.index(r) for r in seq], dtype=torch.long, device=device)

	# Different dihedral potentials for each residue and predicted secondary structure type
	inters_dih = torch.tensor([aas.index(r) * len(ss_types) + ss_types.index(s) for r, s in zip(seq, ss_pred)],
								dtype=torch.long, device=device)



	return native_coords, inters_ang, inters_dih, masses, seq

# TODO ADD DEEPMIND PAPER REFERENCE
def _compute_connectivity(positions, radius, add_self_edges):
	"""Get the indices of connected edges with radius connectivity.
	Args:
	positions: Positions of nodes in the graph. Shape:
		[num_nodes_in_graph, num_dims].
	radius: Radius of connectivity.
	add_self_edges: Whether to include self edges or not.
	Returns:
	senders indices [num_edges_in_graph]
	receiver indices [num_edges_in_graph]
	"""

	tree = neighbors.KDTree(positions)
	receivers_list = tree.query_radius(positions, r=radius)
	num_nodes = len(positions)
	senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
	receivers = np.concatenate(receivers_list, axis=0)

	if not add_self_edges:
		# Remove self edges.
		mask = senders != receivers
		senders = senders[mask]
		receivers = receivers[mask]

	return senders, receivers

def construct_graph_networkx(fp, radius, add_self_edges):

	native_coords, inters_ang, inters_dih, masses, seq = read_input_file(fp)
	senders, receivers = _compute_connectivity(native_coords, radius, add_self_edges)

	G = nx.Graph()

	for i in range(len(masses)):
		G.add_node(i, pos=native_coords[i], mass=masses[i])
	
	G.add_edges_from(zip(senders, receivers))

	atoms_types = []

	for aa in seq:
		for atom in atoms:
			atoms_types.append(f"{aa}_{atom}")

	return G

def construct_graph(fp):
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

	node_f = torch.cat([one_hot_atoms, one_hot_seq, res_numbers], dim=1)


	G = Data(node_f=node_f, pos=native_coords, seq=seq, mass=masses)

	return G


# TODO add RELU
class MLP(nn.Module):
	"""Builds an MLP."""
	def __init__(self, input_size: int, hidden_size: int, num_hidden_layers: int, output_size: int):
		super(MLP, self).__init__()

		layers = [nn.Linear(input_size, hidden_size)]

		for i in range(num_hidden_layers - 1):
			layers.append(nn.Linear(hidden_size, hidden_size))
		
		layers.append(nn.Linear(hidden_size, output_size))
		self.layers = layers

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x

def MLP_with_layernorm(input_size: int, hidden_size: int, 
						num_hidden_layers: int, output_size: int):
	return nn.Sequential(MLP(input_size, hidden_size, 
						num_hidden_layers, output_size),
						nn.LayerNorm())

def knn(P, k):
	"""
	Finds the k-nearest neibours
	"""

	coords = P.pos
	residue_numbers = P.node_f[:,-1]


	N, D = coords.shape
	xyz_i = LazyTensor(coords[:, None, :])
	xyz_j = LazyTensor(coords[None, :, :])

	pairwise_distance_ij = ((xyz_i - xyz_j) ** 2).sum(-1)
	#pairwise_distance_ij.ranges = diagonal_ranges(x_batch, y_batch)

	idx = pairwise_distance_ij.argKmin(K=k, axis=1)  # (N, K)
	x_ik = coords[idx.view(-1)].view(N, k, D)
	distplacements = (coords[:, None, :] - x_ik).view(N*k, 3)
	dists = distplacements.norm(dim=-1).view(N*k,1)

	senders = idx[:,0].repeat(k)
	recievers =  idx.view(N*k)

	edges = torch.cat([senders, recievers]).view(N*k,2)

	res_dists = (residue_numbers[senders] - residue_numbers[recievers]).view(N*k,1)
	
	edge_attr = torch.cat([dists, distplacements, res_dists], dim=1)
	
	return edges, edge_attr
#print(construct_graph("protein_data/example/1CRN.txt", 5 ,False))
#print(construct_features("protein_data/example/1CRN.txt", 5))

def knn_coords(coords, k):
	"""
	Finds the k-nearest neibours
	"""
	coords = coords.to('cpu')
	print(coords)

	B, N, D = coords.shape
	xyz_i = LazyTensor(coords[:, None, :])
	xyz_j = LazyTensor(coords[None, :, :])

	pairwise_distance_ij = ((xyz_i - xyz_j) ** 2).sum(-1)

	idx = pairwise_distance_ij.argKmin(K=k, axis=1)  # (N, K)
	x_ik = coords[idx.view(-1)].view(N, k, D)
	distplacements = (coords[:, None, :] - x_ik).view(N*k, 3)
	dists = distplacements.norm(dim=-1).view(N*k,1)

	senders = idx[:,0].repeat(k)
	recievers =  idx.view(N*k)

	edges = torch.cat([senders, recievers]).view(N*k,2)

	return edges, dists


def rmsd(c1, c2):
	device = c1.device
	r1 = c1.transpose(0, 1)
	r2 = c2.transpose(0, 1)
	P = r1 - r1.mean(1).view(3, 1)
	Q = r2 - r2.mean(1).view(3, 1)
	cov = torch.matmul(P, Q.transpose(0, 1))
	try:
		U, S, V = torch.svd(cov)
	except RuntimeError:
		report("  SVD failed to converge", 0)
		return torch.tensor([20.0], device=device), False
	d = torch.tensor([
			[1.0, 0.0, 0.0],
			[0.0, 1.0, 0.0],
			[0.0, 0.0, torch.det(torch.matmul(V, U.transpose(0, 1)))]
	], device=device)
	rot = torch.matmul(torch.matmul(V, d), U.transpose(0, 1))
	rot_P = torch.matmul(rot, P)
	diffs = rot_P - Q
	msd = (diffs ** 2).sum() / diffs.size(1)
	return msd.sqrt(), True