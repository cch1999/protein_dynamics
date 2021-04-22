# Differentiable molecular simulation of proteins with a coarse-grained potential


class EncodeProcessDecode(torch.nn.Module):
    def __init__(self, ff_distances, ff_angles, ff_dihedrals):
        super(Simulator, self).__init__()
        self.ff_distances = torch.nn.Parameter(ff_distances)
        self.ff_angles    = torch.nn.Parameter(ff_angles)
        self.ff_dihedrals = torch.nn.Parameter(ff_dihedrals)

    def forward(self, input_graph: gn.graphs.GraphsTuple) -> tf.Tensor:
        """Forward pass of the learnable dynamics model."""

        # Encode the input_graph.
        latent_graph_0 = self._encode(input_graph)

        # Do `m` message passing steps in the latent graphs.
        latent_graph_m = self._process(latent_graph_0)

        # Decode from the last latent graph.
        return self._decode(latent_graph_m)        
 
    def _encode(self):
    def _process(self):
    def _decode(self):

class LearnedSimulator(torch.nn.Module):
    def __init__(self, ff_distances, ff_angles, ff_dihedrals):
        super(Simulator, self).__init__()

        self.simulator = LearnedSimulator()

    def forward(self, coords, n_steps, timestep):

        self._encoder_preprocessor(coords)

        for i in range(n_steps):

            self.simulator()
            self.update()

            vels = vels + 0.5 * (accs_last + accs) * timestep
            accs_last = accs

    def _encoder_preprocessor(self):

    def _update(self, normalized_acceleration, position_sequence):
        # The model produces the output in normalized space so we apply inverse
        # normalization.
        acceleration_stats = self._normalization_stats["acceleration"]
        acceleration = (
            normalized_acceleration * acceleration_stats.std
            ) + acceleration_stats.mean

        # Use an Euler integrator to go from acceleration to position, assuming
        # a dt=1 corresponding to the size of the finite difference.
        most_recent_position = position_sequence[:, -1]
        most_recent_velocity = most_recent_position - position_sequence[:, -2]

        new_velocity = most_recent_velocity + acceleration  # * dt = 1
        new_position = most_recent_position + new_velocity  # * dt = 1
        return new_position


class Simulator(torch.nn.Module):
    def __init__(self, ff_distances, ff_angles, ff_dihedrals):
        super(Simulator, self).__init__()
        self.ff_distances = torch.nn.Parameter(ff_distances)
        self.ff_angles    = torch.nn.Parameter(ff_angles)
        self.ff_dihedrals = torch.nn.Parameter(ff_dihedrals)

    def forward(self, coords, n_steps, timestep):

        for i in range(n_steps):
            coords = coords + vels * timestep + 0.5 * accs_last * timestep * timestep

            self._distance_forces()
            self._angle_forces()
            self._dihedral_forces()

            vels = vels + 0.5 * (accs_last + accs) * timestep
            accs_last = accs

    def _distance_forces(self):
    def _angle_forces(self):
    def _dihedral_forces(self):



# Differentiable molecular simulation of proteins with a coarse-grained potential
class Simulator(torch.nn.Module):
    def __init__(self, ff_distances, ff_angles, ff_dihedrals):
        super(Simulator, self).__init__()
        self.ff_distances = torch.nn.Parameter(ff_distances)
        self.ff_angles    = torch.nn.Parameter(ff_angles)
        self.ff_dihedrals = torch.nn.Parameter(ff_dihedrals)

    def forward(self,
                coords,
                inters_flat,
                inters_ang,
                inters_dih,
                masses,
                seq,
                native_coords,
                n_steps,
                integrator="vel", # vel/no_vel/min/langevin/langevin_simple
                timestep=0.02,
                start_temperature=0.1,
                thermostat_const=0.0, # Set to 0.0 to run without a thermostat (NVE ensemble)
                temperature=0.0, # The effective temperature of the thermostat
                sim_filepath=None, # Output PDB file to write to or None to not write out
                energy=False, # Return the energy at the end of the simulation
                report_n=10_000, # Print and write PDB every report_n steps
                verbosity=2, # 0 for epoch info, 1 for protein info, 2 for simulation step info
        ):

        assert integrator in ("vel", "no_vel", "min", "langevin", "langevin_simple"), f"Invalid integrator {integrator}"
        device = coords.device
        batch_size, n_atoms = masses.size(0), masses.size(1)
        n_res = n_atoms // len(atoms)
        dist_bin_centres_tensor = torch.tensor(dist_bin_centres, device=device)
        pair_centres_flat = dist_bin_centres_tensor.index_select(0, inters_flat[0]).unsqueeze(0).expand(batch_size, -1, -1)
        pair_pots_flat = self.ff_distances.index_select(0, inters_flat[0]).unsqueeze(0).expand(batch_size, -1, -1)
        angle_bin_centres_tensor = torch.tensor(angle_bin_centres, device=device)
        angle_centres_flat = angle_bin_centres_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, n_res, -1)
        angle_pots_flat = self.ff_angles.index_select(1, inters_ang[0]).unsqueeze(0).expand(batch_size, -1, -1, -1)
        dih_bin_centres_tensor = torch.tensor(dih_bin_centres, device=device)
        dih_centres_flat = dih_bin_centres_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, n_res - 1, -1)
        dih_pots_flat = self.ff_dihedrals.index_select(1, inters_dih[0]).unsqueeze(0).expand(batch_size, -1, -1, -1)
        native_coords_ca = native_coords.view(batch_size, n_res, 3 * len(atoms))[0, :, 3:6]
        model_n = 0

        if integrator == "vel" or integrator == "langevin" or integrator == "langevin_simple":
            vels = torch.randn(coords.shape, device=device) * start_temperature
            accs_last = torch.zeros(coords.shape, device=device)
        elif integrator == "no_vel":
            coords_last = coords.clone() + torch.randn(coords.shape, device=device) * start_temperature * timestep

        # The step the energy is return on is not used for simulation so we add an extra step
        if energy:
            n_steps += 1

        for i in range(n_steps):
            if integrator == "vel":
                coords = coords + vels * timestep + 0.5 * accs_last * timestep * timestep
            elif integrator == "langevin":
                # From Gronbech-Jensen 2013
                alpha, twokbT = thermostat_const, temperature
                beta = np.sqrt(twokbT * alpha * timestep) * torch.randn(vels.shape, device=device)
                b = 1.0 / (1.0 + (alpha * timestep) / (2 * masses.unsqueeze(2)))
                coords_last = coords
                coords = coords + b * timestep * vels + 0.5 * b * (timestep ** 2) * accs_last + 0.5 * b * timestep * beta / masses.unsqueeze(2)
            elif integrator == "langevin_simple":
                coords = coords + vels * timestep + 0.5 * accs_last * timestep * timestep

            # See https://arxiv.org/pdf/1401.1181.pdf for derivation of forces
            printing = verbosity >= 2 and i % report_n == 0
            returning_energy = energy and i == n_steps - 1
            if printing or returning_energy:
                dist_energy = torch.zeros(1, device=device)
                angle_energy = torch.zeros(1, device=device)
                dih_energy = torch.zeros(1, device=device)

            # Add pairwise distance forces
            crep = coords.unsqueeze(1).expand(-1, n_atoms, -1, -1)
            diffs = crep - crep.transpose(1, 2)
            dists = diffs.norm(dim=3)
            dists_flat = dists.view(batch_size, n_atoms * n_atoms)
            dists_from_centres = pair_centres_flat - dists_flat.unsqueeze(2).expand(-1, -1, n_bins_force)
            dist_bin_inds = dists_from_centres.abs().argmin(dim=2).unsqueeze(2)
            # Force is gradient of potential
            # So it is proportional to difference of previous and next value of potential
            pair_forces_flat = 0.5 * (pair_pots_flat.gather(2, dist_bin_inds) - pair_pots_flat.gather(2, dist_bin_inds + 2))
            # Specify minimum to prevent division by zero errors
            norm_diffs = diffs / dists.clamp(min=0.01).unsqueeze(3)
            pair_accs = (pair_forces_flat.view(batch_size, n_atoms, n_atoms)).unsqueeze(3) * norm_diffs
            accs = pair_accs.sum(dim=1) / masses.unsqueeze(2)
            if printing or returning_energy:
                dist_energy += 0.5 * pair_pots_flat.gather(2, dist_bin_inds + 1).sum()

            atom_coords = coords.view(batch_size, n_res, 3 * len(atoms))
            atom_accs = torch.zeros(batch_size, n_res, 3 * len(atoms), device=device)
            # Angle forces
            # across_res is the number of atoms in the next residue, starting from atom_3
            for ai, (atom_1, atom_2, atom_3, across_res) in enumerate(angles):
                ai_1, ai_2, ai_3 = atoms.index(atom_1), atoms.index(atom_2), atoms.index(atom_3)
                if across_res == 0:
                    ba = atom_coords[:, :  , (ai_1 * 3):(ai_1 * 3 + 3)] - atom_coords[:, :  , (ai_2 * 3):(ai_2 * 3 + 3)]
                    bc = atom_coords[:, :  , (ai_3 * 3):(ai_3 * 3 + 3)] - atom_coords[:, :  , (ai_2 * 3):(ai_2 * 3 + 3)]
                    # Use residue potential according to central atom
                    angle_pots_to_use = angle_pots_flat[:, ai, :]
                elif across_res == 1:
                    ba = atom_coords[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] - atom_coords[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)]
                    bc = atom_coords[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] - atom_coords[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)]
                    angle_pots_to_use = angle_pots_flat[:, ai, :-1]
                elif across_res == 2:
                    ba = atom_coords[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] - atom_coords[:, 1: , (ai_2 * 3):(ai_2 * 3 + 3)]
                    bc = atom_coords[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] - atom_coords[:, 1: , (ai_2 * 3):(ai_2 * 3 + 3)]
                    angle_pots_to_use = angle_pots_flat[:, ai, 1:]
                ba_norms = ba.norm(dim=2)
                bc_norms = bc.norm(dim=2)
                angs = torch.acos((ba * bc).sum(dim=2) / (ba_norms * bc_norms))
                n_angles = n_res if across_res == 0 else n_res - 1
                angles_from_centres = angle_centres_flat[:, :n_angles] - angs.unsqueeze(2)
                angle_bin_inds = angles_from_centres.abs().argmin(dim=2).unsqueeze(2)
                angle_forces = 0.5 * (angle_pots_to_use.gather(2, angle_bin_inds) - angle_pots_to_use.gather(2, angle_bin_inds + 2))
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
                if printing or returning_energy:
                    angle_energy += angle_pots_to_use.gather(2, angle_bin_inds + 1).sum()

            # Dihedral forces
            # across_res is the number of atoms in the next residue, starting from atom_4
            for di, (atom_1, atom_2, atom_3, atom_4, across_res) in enumerate(dihedrals):
                ai_1, ai_2, ai_3, ai_4 = atoms.index(atom_1), atoms.index(atom_2), atoms.index(atom_3), atoms.index(atom_4)
                if across_res == 1:
                    ab = atom_coords[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)] - atom_coords[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)]
                    bc = atom_coords[:, :-1, (ai_3 * 3):(ai_3 * 3 + 3)] - atom_coords[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)]
                    cd = atom_coords[:, 1: , (ai_4 * 3):(ai_4 * 3 + 3)] - atom_coords[:, :-1, (ai_3 * 3):(ai_3 * 3 + 3)]
                    # Use residue potential according to central atom
                    dih_pots_to_use = dih_pots_flat[:, di, :-1]
                elif across_res == 2:
                    ab = atom_coords[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)] - atom_coords[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)]
                    bc = atom_coords[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] - atom_coords[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)]
                    cd = atom_coords[:, 1: , (ai_4 * 3):(ai_4 * 3 + 3)] - atom_coords[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)]
                    dih_pots_to_use = dih_pots_flat[:, di, 1:]
                elif across_res == 3:
                    ab = atom_coords[:, 1: , (ai_2 * 3):(ai_2 * 3 + 3)] - atom_coords[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)]
                    bc = atom_coords[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] - atom_coords[:, 1: , (ai_2 * 3):(ai_2 * 3 + 3)]
                    cd = atom_coords[:, 1: , (ai_4 * 3):(ai_4 * 3 + 3)] - atom_coords[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)]
                    dih_pots_to_use = dih_pots_flat[:, di, 1:]
                cross_ab_bc = torch.cross(ab, bc, dim=2)
                cross_bc_cd = torch.cross(bc, cd, dim=2)
                bc_norms = bc.norm(dim=2).unsqueeze(2)
                dihs = torch.atan2(
                    torch.sum(torch.cross(cross_ab_bc, cross_bc_cd, dim=2) * bc / bc_norms, dim=2),
                    torch.sum(cross_ab_bc * cross_bc_cd, dim=2)
                )
                dihs_from_centres = dih_centres_flat - dihs.unsqueeze(2)
                dih_bin_inds = dihs_from_centres.abs().argmin(dim=2).unsqueeze(2)
                dih_forces = 0.5 * (dih_pots_to_use.gather(2, dih_bin_inds) - dih_pots_to_use.gather(2, dih_bin_inds + 2))
                fa = dih_forces * normalize(-cross_ab_bc, dim=2) / ab.norm(dim=2).unsqueeze(2)
                fd = dih_forces * normalize( cross_bc_cd, dim=2) / cd.norm(dim=2).unsqueeze(2)
                # Forces on the middle atoms have to keep the sum of torques null
                # Forces taken from http://www.softberry.com/freedownloadhelp/moldyn/description.html
                fb = ((ab * -bc) / (bc_norms ** 2) - 1) * fa - ((cd * -bc) / (bc_norms ** 2)) * fd
                fc = -fa - fb - fd
                if across_res == 1:
                    atom_accs[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] += fa
                    atom_accs[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)] += fb
                    atom_accs[:, :-1, (ai_3 * 3):(ai_3 * 3 + 3)] += fc
                    atom_accs[:, 1: , (ai_4 * 3):(ai_4 * 3 + 3)] += fd
                elif across_res == 2:
                    atom_accs[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] += fa
                    atom_accs[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)] += fb
                    atom_accs[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] += fc
                    atom_accs[:, 1: , (ai_4 * 3):(ai_4 * 3 + 3)] += fd
                elif across_res == 3:
                    atom_accs[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] += fa
                    atom_accs[:, 1: , (ai_2 * 3):(ai_2 * 3 + 3)] += fb
                    atom_accs[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] += fc
                    atom_accs[:, 1: , (ai_4 * 3):(ai_4 * 3 + 3)] += fd
                if printing or returning_energy:
                    dih_energy += dih_pots_to_use.gather(2, dih_bin_inds + 1).sum()

            accs += atom_accs.view(batch_size, n_atoms, 3) / masses.unsqueeze(2)

            # Shortcut to return energy at a given step
            if returning_energy:
                return dist_energy + angle_energy + dih_energy

            if integrator == "vel":
                vels = vels + 0.5 * (accs_last + accs) * timestep
                accs_last = accs
            elif integrator == "no_vel":
                coords_next = 2 * coords - coords_last + accs * timestep * timestep
                coords_last = coords
                coords = coords_next
            elif integrator == "langevin":
                # From Gronbech-Jensen 2013
                vels = vels + 0.5 * timestep * (accs_last + accs) - alpha * (coords - coords_last) / masses.unsqueeze(2) + beta / masses.unsqueeze(2)
                accs_last = accs
            elif integrator == "langevin_simple":
                gamma, twokbT = thermostat_const, temperature
                accs = accs + (-gamma * vels + np.sqrt(gamma * twokbT) * torch.randn(vels.shape, device=device)) / masses.unsqueeze(2)
                vels = vels + 0.5 * (accs_last + accs) * timestep
                accs_last = accs
            elif integrator == "min":
                coords = coords + accs * 0.1

            # Apply thermostat
            if integrator in ("vel", "no_vel") and thermostat_const > 0.0:
                thermostat_prob = timestep / thermostat_const
                for ai in range(n_atoms):
                    if random() < thermostat_prob:
                        if integrator == "vel":
                            # Actually this should be divided by the mass
                            new_vel = torch.randn(3, device=device) * temperature
                            vels[0, ai] = new_vel
                        elif integrator == "no_vel":
                            new_diff = torch.randn(3, device=device) * temperature * timestep
                            coords_last[0, ai] = coords[0, ai] - new_diff

            if printing:
                total_energy = dist_energy + angle_energy + dih_energy
                out_line = "    Step {:8} / {} - acc {:6.3f} {}- energy {:6.2f} ( {:6.2f} {:6.2f} {:6.2f} ) - Cα RMSD {:6.2f}".format(
                    i + 1, n_steps, torch.mean(accs.norm(dim=2)).item(),
                    "- vel {:6.3f} ".format(torch.mean(vels.norm(dim=2)).item()) if integrator in ("vel", "langevin", "langevin_simple") else "",
                    total_energy.item(), dist_energy.item(), angle_energy.item(), dih_energy.item(),
                    rmsd(coords.view(batch_size, n_res, 3 * len(atoms))[0, :, 3:6], native_coords_ca)[0].item())
                report(out_line, 2, verbosity)

            if sim_filepath and i % report_n == 0:
                model_n += 1
                with open(sim_filepath, "a") as of:
                    of.write("MODEL {:>8}\n".format(model_n))
                    for ri, r in enumerate(seq):
                        for ai, atom in enumerate(atoms):
                            of.write("ATOM   {:>4}  {:<2}  {:3} A{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}  1.00  0.00          {:>2}  \n".format(
                                len(atoms) * ri + ai + 1, atom[:2].upper(),
                                one_to_three_aas[r], ri + 1,
                                coords[0, len(atoms) * ri + ai, 0].item(),
                                coords[0, len(atoms) * ri + ai, 1].item(),
                                coords[0, len(atoms) * ri + ai, 2].item(),
                                atom[0].upper()))
                    of.write("ENDMDL\n")

        return coords