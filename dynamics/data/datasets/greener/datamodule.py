import os
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset

from dynamics.data.datasets.greener.variables import *

class GreenerDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.setup() # TODO Check way this is not running on init

    def setup(self, stage = 'train'):

        # COPY AND PASTA
        train_proteins = [l.rstrip() for l in open(os.path.join(self.data_dir, "splits/train.txt"))]
        val_proteins   = [l.rstrip() for l in open(os.path.join(self.data_dir, "splits/val.txt"  ))]

        coords_dir = os.path.join(self.data_dir, "data/coords/")

        self.train = ProteinDataset(train_proteins, coords_dir) # TODO FIX DIRS
        self.test = ProteinDataset(val_proteins, coords_dir)


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


class ProteinDataset(Dataset):
    def __init__(self, pdbids, coord_dir):
        self.pdbids = pdbids
        self.coord_dir = coord_dir
        self.set_size = len(pdbids)

    def __len__(self):
        return self.set_size

    def __getitem__(self, index):
        fp = os.path.join(self.coord_dir, self.pdbids[index] + ".txt")
        return get_features(fp)

def get_features(fp):

	# TODO remove inters constructions
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

if __name__ == "__main__":



    dm = GreenerDataModule("/Users/charlie/projects/protein_dynamics/dynamics/data/datasets/greener", 1)

    val_loader = dm.val_dataloader()

    print(len(val_loader))

    print(next(iter(val_loader)))

    print('Done')