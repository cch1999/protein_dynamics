import torch
import torch.nn as nn
from torch.nn.functional import normalize


from dynamics.model.layers.linear import MLP, ResNet
from dynamics.data.datasets.greener.variables import atoms, angles, dihedrals
from dynamics.model.utils.geometric import knn


class DistanceForces(nn.Module):
    """
    Calculates forces between two atoms based on their
            1. atoms types
            2. Euclidian distance
            3. Seperation along the sequence

    Input dim = 50 (24*2 + 2)
    Output dim = 1 (a scalar force)
    """

    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size):
        super(DistanceForces, self).__init__()

        self.model = ResNet(input_size, hidden_size, num_hidden_layers, output_size)

    def forward(self, atom1, atom2, edges):
        messages = torch.cat([atom1 + atom2, edges], dim=1)
        return self.model(messages)


class AngleForces(nn.Module):
    """
    Calculates forces between three atoms making an angle on their
            1. central atom types
            2. angle around the central atom

    Input dim = 25 (24 + 1)
    Output dim = 1 (a scalar force)
    """

    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size):
        super(AngleForces, self).__init__()

        self.model = ResNet(input_size, hidden_size, num_hidden_layers, output_size)

    def forward(self, atom1, atom2, atom3, angles):
        messages = torch.cat([atom1, atom2, atom3, angles[:, :, None]], dim=2)
        return self.model(messages)


class DihedralForces(nn.Module):
    """
    Calculates forces between three atoms making an angle on their
            1. central atom types
            2. angle around the central atom

    Input dim = 25 (24 + 1)
    Output dim = 1 (a scalar force)
    """

    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size):
        super(DihedralForces, self).__init__()

        self.model = ResNet(input_size, hidden_size, num_hidden_layers, output_size)

    def forward(self, atom1, atom2, atom3, atom4, dihedrals):
        messages = torch.cat([atom1, atom2, atom3, atom4, dihedrals[:, :, None]], dim=2)
        return self.model(messages)


# ----Main model----#
class PBMP(nn.Module):
    def __init__(self, temperature, timestep, n_steps, k):
        super().__init__()

        self.temperature = temperature
        self.timestep = timestep
        self.n_steps = n_steps
        self.k = k

        self.distance_forces = DistanceForces(24 + 2, 100, 15, 1)
        self.angle_forces = AngleForces(24 * 3 + 1, 128, 3, 1)
        self.dihedral_forces = DihedralForces(24 * 4 + 1, 128, 5, 1)

    def forward(self, P):
        """
        coords, node_f, res_numbers, masses, seq,
                k, n_steps, timestep, temperature, animation, device):
        """

        animation = False

        k = self.k
        temperature = self.temperature
        n_steps = self.n_steps
        timestep = self.timestep

        coords, vels, accs_last, node_f = P.pos, P.vels, P.accs_last, P.x

        n_atoms = coords.shape[0]
        n_res = n_atoms // len(atoms)
        model_n = 0

        coords, vels, res_numbers, masses, seq = (
            P.pos,
            P.vels,
            P.res_numbers,
            P.masses,
            P.seq,
        )

        for i in range(n_steps):

            coords = coords + vels * timestep + 0.5 * accs_last * timestep * timestep

            idx = knn(coords, k + 1)
            senders = idx[:, 0].repeat_interleave(k)
            receivers = idx[:, 1:].reshape(n_atoms * k)

            # Calc Euclidian distance
            diffs = coords[senders] - coords[receivers]
            dists = diffs.norm(dim=1)
            norm_diffs = diffs / dists.clamp(min=0.01).unsqueeze(1)

            # Calc sequence seperation
            seq_sep = abs(res_numbers[senders] - res_numbers[receivers]) / 5
            mask = seq_sep > 1
            seq_sep[mask] = 1

            # Concat edge features
            edges1 = torch.cat([dists.unsqueeze(1) - 0.01, seq_sep], dim=1)
            edges2 = torch.cat([dists.unsqueeze(1) + 0.01, seq_sep], dim=1)

            # Compute forces using MLP
            forces = 50 * (
                self.distance_forces(node_f[senders], node_f[receivers], edges1)
                - self.distance_forces(node_f[senders], node_f[receivers], edges2)
            )

            forces = forces * norm_diffs
            total_forces = forces.view(n_atoms, k, 3).sum(1)

            batch_size = 1
            atom_types = node_f.view(batch_size, n_res, len(atoms), 24)
            atom_coords = coords.view(batch_size, n_res, 3 * len(atoms))
            atom_accs = torch.zeros(batch_size, n_res, 3 * len(atoms))
            # Angle forces
            # across_res is the number of atoms in the next residue, starting from atom_3
            for ai, (atom_1, atom_2, atom_3, across_res) in enumerate(angles):
                # Calc vectors and angle between atoms
                ai_1, ai_2, ai_3 = (
                    atoms.index(atom_1),
                    atoms.index(atom_2),
                    atoms.index(atom_3),
                )
                if across_res == 0:
                    ba = (
                        atom_coords[:, :, (ai_1 * 3) : (ai_1 * 3 + 3)]
                        - atom_coords[:, :, (ai_2 * 3) : (ai_2 * 3 + 3)]
                    )
                    bc = (
                        atom_coords[:, :, (ai_3 * 3) : (ai_3 * 3 + 3)]
                        - atom_coords[:, :, (ai_2 * 3) : (ai_2 * 3 + 3)]
                    )
                elif across_res == 1:
                    ba = (
                        atom_coords[:, :-1, (ai_1 * 3) : (ai_1 * 3 + 3)]
                        - atom_coords[:, :-1, (ai_2 * 3) : (ai_2 * 3 + 3)]
                    )
                    bc = (
                        atom_coords[:, 1:, (ai_3 * 3) : (ai_3 * 3 + 3)]
                        - atom_coords[:, :-1, (ai_2 * 3) : (ai_2 * 3 + 3)]
                    )
                elif across_res == 2:
                    ba = (
                        atom_coords[:, :-1, (ai_1 * 3) : (ai_1 * 3 + 3)]
                        - atom_coords[:, 1:, (ai_2 * 3) : (ai_2 * 3 + 3)]
                    )
                    bc = (
                        atom_coords[:, 1:, (ai_3 * 3) : (ai_3 * 3 + 3)]
                        - atom_coords[:, 1:, (ai_2 * 3) : (ai_2 * 3 + 3)]
                    )
                ba_norms = ba.norm(dim=2)
                bc_norms = bc.norm(dim=2)
                angs = torch.acos((ba * bc).sum(dim=2) / (ba_norms * bc_norms))
                # Get central atom properties
                if ai == 0:
                    atom1 = atom_types[:, :, 0, :]
                    atom2 = atom_types[:, :, 1, :]
                    atom3 = atom_types[:, :, 2, :]
                if ai == 1:
                    atom1 = atom_types[:, :-1, 1, :]
                    atom2 = atom_types[:, :-1, 2, :]
                    atom3 = atom_types[:, 1:, 0, :]
                if ai == 2:
                    atom1 = atom_types[:, :-1, 2, :]
                    atom2 = atom_types[:, 1:, 0, :]
                    atom1 = atom_types[:, 1:, 1, :]
                if ai == 3:
                    atom1 = atom_types[:, :, 0, :]
                    atom2 = atom_types[:, :, 1, :]
                    atom3 = atom_types[:, :, 3, :]
                if ai == 4:
                    atom1 = atom_types[:, :, 2, :]
                    atom2 = atom_types[:, :, 1, :]
                    atom3 = atom_types[:, :, 3, :]

                angle_forces = 50 * (
                    self.angle_forces(atom1, atom2, atom3, angs - 0.01)
                    - self.angle_forces(atom1, atom2, atom3, angs + 0.01)
                )

                cross_ba_bc = torch.cross(ba, bc, dim=2)
                fa = (
                    angle_forces
                    * normalize(torch.cross(ba, cross_ba_bc, dim=2), dim=2)
                    / ba_norms.unsqueeze(2)
                )
                fc = (
                    angle_forces
                    * normalize(torch.cross(-bc, cross_ba_bc, dim=2), dim=2)
                    / bc_norms.unsqueeze(2)
                )
                fb = -fa - fc
                if across_res == 0:
                    atom_accs[:, :, (ai_1 * 3) : (ai_1 * 3 + 3)] += fa
                    atom_accs[:, :, (ai_2 * 3) : (ai_2 * 3 + 3)] += fb
                    atom_accs[:, :, (ai_3 * 3) : (ai_3 * 3 + 3)] += fc
                elif across_res == 1:
                    atom_accs[:, :-1, (ai_1 * 3) : (ai_1 * 3 + 3)] += fa
                    atom_accs[:, :-1, (ai_2 * 3) : (ai_2 * 3 + 3)] += fb
                    atom_accs[:, 1:, (ai_3 * 3) : (ai_3 * 3 + 3)] += fc
                elif across_res == 2:
                    atom_accs[:, :-1, (ai_1 * 3) : (ai_1 * 3 + 3)] += fa
                    atom_accs[:, 1:, (ai_2 * 3) : (ai_2 * 3 + 3)] += fb
                    atom_accs[:, 1:, (ai_3 * 3) : (ai_3 * 3 + 3)] += fc

                    # Dihedral forces
            # across_res is the number of atoms in the next residue, starting from atom_4
            for di, (atom_1, atom_2, atom_3, atom_4, across_res) in enumerate(
                dihedrals
            ):
                ai_1, ai_2, ai_3, ai_4 = (
                    atoms.index(atom_1),
                    atoms.index(atom_2),
                    atoms.index(atom_3),
                    atoms.index(atom_4),
                )
                if across_res == 1:
                    ab = (
                        atom_coords[:, :-1, (ai_2 * 3) : (ai_2 * 3 + 3)]
                        - atom_coords[:, :-1, (ai_1 * 3) : (ai_1 * 3 + 3)]
                    )
                    bc = (
                        atom_coords[:, :-1, (ai_3 * 3) : (ai_3 * 3 + 3)]
                        - atom_coords[:, :-1, (ai_2 * 3) : (ai_2 * 3 + 3)]
                    )
                    cd = (
                        atom_coords[:, 1:, (ai_4 * 3) : (ai_4 * 3 + 3)]
                        - atom_coords[:, :-1, (ai_3 * 3) : (ai_3 * 3 + 3)]
                    )
                elif across_res == 2:
                    ab = (
                        atom_coords[:, :-1, (ai_2 * 3) : (ai_2 * 3 + 3)]
                        - atom_coords[:, :-1, (ai_1 * 3) : (ai_1 * 3 + 3)]
                    )
                    bc = (
                        atom_coords[:, 1:, (ai_3 * 3) : (ai_3 * 3 + 3)]
                        - atom_coords[:, :-1, (ai_2 * 3) : (ai_2 * 3 + 3)]
                    )
                    cd = (
                        atom_coords[:, 1:, (ai_4 * 3) : (ai_4 * 3 + 3)]
                        - atom_coords[:, 1:, (ai_3 * 3) : (ai_3 * 3 + 3)]
                    )
                elif across_res == 3:
                    ab = (
                        atom_coords[:, 1:, (ai_2 * 3) : (ai_2 * 3 + 3)]
                        - atom_coords[:, :-1, (ai_1 * 3) : (ai_1 * 3 + 3)]
                    )
                    bc = (
                        atom_coords[:, 1:, (ai_3 * 3) : (ai_3 * 3 + 3)]
                        - atom_coords[:, 1:, (ai_2 * 3) : (ai_2 * 3 + 3)]
                    )
                    cd = (
                        atom_coords[:, 1:, (ai_4 * 3) : (ai_4 * 3 + 3)]
                        - atom_coords[:, 1:, (ai_3 * 3) : (ai_3 * 3 + 3)]
                    )
                if di == 0:
                    atom1 = atom_types[:, :-1, 2, :]
                    atom2 = atom_types[:, 1:, 0, :]
                    atom3 = atom_types[:, 1:, 1, :]
                    atom4 = atom_types[:, 1:, 2, :]
                if di == 1:
                    atom1 = atom_types[:, :-1, 0, :]
                    atom2 = atom_types[:, :-1, 1, :]
                    atom3 = atom_types[:, :-1, 2, :]
                    atom4 = atom_types[:, 1:, 0, :]
                if di == 2:
                    atom1 = atom_types[:, :-1, 1, :]
                    atom2 = atom_types[:, :-1:, 2, :]
                    atom3 = atom_types[:, 1:, 0, :]
                    atom4 = atom_types[:, 1:, 1, :]
                if di == 3:
                    atom1 = atom_types[:, :-1, 2, :]
                    atom2 = atom_types[:, 1:, 0, :]
                    atom3 = atom_types[:, 1:, 1, :]
                    atom4 = atom_types[:, 1:, 3, :]
                if di == 4:
                    atom1 = atom_types[:, :-1, 3, :]
                    atom2 = atom_types[:, :-1, 1, :]
                    atom3 = atom_types[:, :-1, 2, :]
                    atom4 = atom_types[:, 1:, 0, :]
                cross_ab_bc = torch.cross(ab, bc, dim=2)
                cross_bc_cd = torch.cross(bc, cd, dim=2)
                bc_norms = bc.norm(dim=2).unsqueeze(2)
                dihs = torch.atan2(
                    torch.sum(
                        torch.cross(cross_ab_bc, cross_bc_cd, dim=2) * bc / bc_norms,
                        dim=2,
                    ),
                    torch.sum(cross_ab_bc * cross_bc_cd, dim=2),
                )

                dih_forces = 50 * (
                    self.dihedral_forces(atom1, atom2, atom3, atom4, dihs - 0.01)
                    - self.dihedral_forces(atom1, atom2, atom3, atom4, dihs + 0.01)
                )

                fa = (
                    dih_forces
                    * normalize(-cross_ab_bc, dim=2)
                    / ab.norm(dim=2).unsqueeze(2)
                )
                fd = (
                    dih_forces
                    * normalize(cross_bc_cd, dim=2)
                    / cd.norm(dim=2).unsqueeze(2)
                )
                # Forces on the middle atoms have to keep the sum of torques null
                # Forces taken from http://www.softberry.com/freedownloadhelp/moldyn/description.html
                fb = ((ab * -bc) / (bc_norms**2) - 1) * fa - (
                    (cd * -bc) / (bc_norms**2)
                ) * fd
                fc = -fa - fb - fd
                if across_res == 1:
                    atom_accs[:, :-1, (ai_1 * 3) : (ai_1 * 3 + 3)] += fa
                    atom_accs[:, :-1, (ai_2 * 3) : (ai_2 * 3 + 3)] += fb
                    atom_accs[:, :-1, (ai_3 * 3) : (ai_3 * 3 + 3)] += fc
                    atom_accs[:, 1:, (ai_4 * 3) : (ai_4 * 3 + 3)] += fd
                elif across_res == 2:
                    atom_accs[:, :-1, (ai_1 * 3) : (ai_1 * 3 + 3)] += fa
                    atom_accs[:, :-1, (ai_2 * 3) : (ai_2 * 3 + 3)] += fb
                    atom_accs[:, 1:, (ai_3 * 3) : (ai_3 * 3 + 3)] += fc
                    atom_accs[:, 1:, (ai_4 * 3) : (ai_4 * 3 + 3)] += fd
                elif across_res == 3:
                    atom_accs[:, :-1, (ai_1 * 3) : (ai_1 * 3 + 3)] += fa
                    atom_accs[:, 1:, (ai_2 * 3) : (ai_2 * 3 + 3)] += fb
                    atom_accs[:, 1:, (ai_3 * 3) : (ai_3 * 3 + 3)] += fc
                    atom_accs[:, 1:, (ai_4 * 3) : (ai_4 * 3 + 3)] += fd

            # Calc distance accs
            accs = total_forces / masses.unsqueeze(1)
            # Calc angle accs
            accs += atom_accs.view(n_atoms, 3) / (masses.unsqueeze(1) * 100)

            vels = vels + 0.5 * (accs_last + accs) * timestep
            accs_last = accs

            if animation:
                if i % animation == 0:
                    model_n += 1
                    save_structure(coords[None, :, :], "animation.pdb", seq, model_n)

        P.coords = coords
        P.vels = vels
        P.accs_last = accs_last
        P.randn_coords = P.native_coords + P.vels * timestep * n_steps

        return P
