import torch
import numpy as np
from model6 import Simulator, get_features
import matplotlib.pyplot as plt

device = "cuda:6"

model = Simulator(50, 128, 1).to(device)
model.load_state_dict(torch.load("models/current.pt", map_location=device))
model.eval()

with torch.no_grad():
    coords, node_f, res_numbers, masses, seq = get_features("protein_data/example/1CRN.txt", device=device)
    
    out, basic_loss = model(coords, node_f, res_numbers, masses, seq, 10, 
                    n_steps=50000, timestep=0.02, temperature=0.02,
                    animation=100, device=device)
    

    """
    model = model.angle_forces

    atom = torch.zeros(24).to(device).double()
    atom[[2,4]] = 1
    print(atom)
    forces = []
    angles = np.arange(0, 6.2, 0.01)
    for i in angles:
        force = model(atom[None,None,:].float(), torch.tensor(i)[None, None].to(device).float())
        print(force)

        forces.append(force.item())
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 4))

    ax1.plot(angles, forces)
    ax1.axhline(0, color='black')
    ax1.axvline(2.02, color='red')
    ax1.set_ylabel("Force")
    ax1.set_xlabel("C Bond angle (radians)")
    ax1.text(2.1, 20, 'C bond angle', color='red')
    ax1.set_title('Force applied to atoms in -C- angle')
    
    pots = []
    y = 0

    for i in range(len(forces)):
        print(i)
        y += -forces[i]
        pots.append(y)
    ax2.plot(angles,pots)
    ax2.axvline(2.02, color='red')
    ax2.set_ylabel("Potential energy")
    ax2.set_xlabel("C Bond angle (radians)")
    ax2.text(2.1, 20, 'C bond angle', color='red')
    ax2.set_title('Potential in -C- angle')
    plt.savefig('c_ang.png')
    """
    """
    # Distances
    model = model.distance_forces
    print(model)

    distances = torch.tensor(np.arange(0,15,0.1)).to(device).float()

    atom = torch.zeros(24).to(device)
    atom1 = atom.clone()
    atom2 = atom.clone()
    atom1[3] = 1
    atom1[14] = 1
    atom2[1] = 1
    atom2[14] = 1
    print(atom1)
    print(atom2)

    forces = []

    for dist in distances:

        force = model(atom1[None,:], atom2[None,:], torch.tensor([dist, 0])[None,:].to(device).float())

        print(force)

        forces.append(force.item())

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 4))

    ax1.plot(distances.cpu(), forces)
    ax1.axhline(0, color='black')
    ax1.axvline(2.1, color='red')
    ax1.set_ylabel("Force")
    ax1.set_xlabel("Bond distance (Å)")
    #ax1.text(1.7, 5, 'True distsance', color='red')
    ax1.set_title('Force between Alanine R-Ca atoms')
    
    pots = []
    y = 0
    print(forces)
    for i in range(len(forces)):
        print(i)
        y += -forces[i]
        pots.append(y)
    print(pots)
    print(len(pots))
    ax2.plot(distances.cpu(), pots)
    ax2.axvline(2.1, color='red')
    ax2.set_ylabel("Potential energy")
    ax2.set_xlabel("Bond distance (Å)")
    #ax2.text(1.7, -20, 'True distance', color='red')
    ax2.set_title('Potential between Alanine R-Ca atoms')
    plt.savefig('I_L_sidechains.png')
 
    """
    """
    # Dihedrals
    model = model.dihedral_forces
    print(model)

    angles = torch.tensor(np.arange(0, 6.2, 0.01), device=device)

    atom = torch.zeros(24).to(device)
    atom1 = atom.clone()
    atom2 = atom.clone()
    atom3 = atom.clone()
    atom4 = atom.clone()
    if False:
        atom1[0] = 1
        atom1[5] = 1

        atom2[1] = 1
        atom2[5] = 1

        atom3[2] = 1
        atom3[5] = 1

        atom4[0] = 1
        atom4[5] = 1

    if True:
        atom1[2] = 1
        atom1[5] = 1

        atom2[0] = 1
        atom2[5] = 1

        atom3[1] = 1
        atom3[5] = 1

        atom4[2] = 1
        atom4[5] = 1
    print(atom1)
    print(atom2)

    forces = []

    for dist in angles:


        force = model(atom1[None,None,:], atom2[None,None,:],
                        atom3[None,None,:], atom4[None,None,:],
                        torch.tensor([dist])[None,:].to(device).float())

        print(force)

        forces.append(force.item())

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 4))

    ax1.plot(angles.cpu(), forces)
    ax1.axhline(0, color='black')
    ax1.axvline(2.1, color='red')
    ax1.set_ylabel("Force")
    ax1.set_xlabel("Bond distance (Å)")
    #ax1.text(1.7, 5, 'True distsance', color='red')
    ax1.set_title('Force between Alanine R-Ca atoms')
    
    pots = []
    y = 0
    print(forces)
    for i in range(len(forces)):
        print(i)
        y += forces[i]
        pots.append(y)
    print(len(pots))
    ax2.plot(angles.cpu(), pots)
    ax2.axvline(2.1, color='red')
    ax2.set_ylabel("Potential energy")
    ax2.set_xlabel("Bond distance (Å)")
    #ax2.text(1.7, -20, 'True distance', color='red')
    ax2.set_title('Potential between Alanine R-Ca atoms')
    plt.savefig('dihedral_angles.png')
    """