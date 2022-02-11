atoms = ["N", "CA", "C", "cent"]

# Last value is the number of atoms in the next residue
angles = [
    ("N", "CA", "C", 0),
    ("CA", "C", "N", 1),
    ("C", "N", "CA", 2),
    ("N", "CA", "cent", 0),
    ("C", "CA", "cent", 0),
]

# Last value is the number of atoms in the next residue
dihedrals = [
    ("C", "N", "CA", "C", 3),
    ("N", "CA", "C", "N", 1),
    ("CA", "C", "N", "CA", 2),
    ("C", "N", "CA", "cent", 3),
    ("cent", "CA", "C", "N", 1),
]

aas = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
n_aas = len(aas)

aa_masses = {
    "A": 89.09,
    "R": 174.2,
    "N": 132.1,
    "D": 133.1,
    "C": 121.2,
    "E": 147.1,
    "Q": 146.1,
    "G": 75.07,
    "H": 155.2,
    "I": 131.2,
    "L": 131.2,
    "K": 146.2,
    "M": 149.2,
    "F": 165.2,
    "P": 115.1,
    "S": 105.09,
    "T": 119.1,
    "W": 204.2,
    "Y": 181.2,
    "V": 117.1,
}

ss_types = ["H", "E", "C"]

# Minima from Greener model
centroid_dists = {
    "A": 1.5575,
    "R": 4.3575,
    "N": 2.5025,
    "D": 2.5025,
    "C": 2.0825,
    "E": 3.3425,
    "Q": 3.3775,
    "G": 1.0325,
    "H": 3.1675,
    "I": 2.3975,
    "L": 2.6075,
    "K": 3.8325,
    "M": 3.1325,
    "F": 3.4125,
    "P": 1.9075,
    "S": 1.9425,
    "T": 1.9425,
    "W": 3.9025,
    "Y": 3.7975,
    "V": 1.9775,
}
