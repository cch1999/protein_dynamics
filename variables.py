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