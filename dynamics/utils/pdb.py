import torch

from dynamics.data.datasets.greener.variables import atoms, one_to_three_aas

def save_structure(coords, sim_filepath, seq, model_n):
	"""
	Save a tradjectory to a PDB file
	"""
	coords = coords.unsqueeze(0)
	
	with open(sim_filepath, "a") as of:
		of.write("MODEL {:>8}\n".format(model_n))
		for ri, r in enumerate(seq[0]):
			for ai, atom in enumerate(atoms):
				of.write("ATOM   {:>4}  {:<2}  {:3} A{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}  1.00  0.00          {:>2}  \n".format(
					len(atoms) * ri + ai + 1, atom[:2].upper(),
					one_to_three_aas[r], ri + 1,
					coords[0, len(atoms) * ri + ai, 0].item(),
					coords[0, len(atoms) * ri + ai, 1].item(),
					coords[0, len(atoms) * ri + ai, 2].item(),
					atom[0].upper()))
		of.write("ENDMDL\n")