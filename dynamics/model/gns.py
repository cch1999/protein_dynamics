import torch
import torch.nn as nn
from torch_geometric.data import Data

from dynamics.model.layers.linear import MLP_with_layer_norm, MLP
from dynamics.model.utils.geometric import knn
from dynamics.utils.pdb import save_structure


class Encoder(nn.Module):
    def __init__(self, node_encoder, edge_encoder):
        super().__init__()

        self.node_encoder = MLP_with_layer_norm(**node_encoder)
        self.edge_encoder = MLP_with_layer_norm(**edge_encoder)

    def forward(self, P):

        P.x = self.node_encoder(P.x_native)
        P.edge_attr = self.edge_encoder(P.edge_attr)

        return P


class MPNN(nn.Module):
    def __init__(self, k, edge_mlp, node_mlp):
        super().__init__()

        self.k = k

        self.edge_mlp = MLP_with_layer_norm(
            edge_mlp.input_size * 3,
            edge_mlp.hidden_size,
            edge_mlp.num_hidden_layers,
            edge_mlp.output_size,
        )

        self.node_mlp = MLP_with_layer_norm(**node_mlp)

    def forward(self, P):
        # Messages
        m = self.edge_mlp(
            torch.cat([P.x[P.receivers], P.x[P.senders], P.edge_attr], -1)
        )

        P.edge_attr = m

        # Aggregate
        x = (m.view(P.n_atoms, 128, self.k)).sum(-1)

        # Update
        P.x = self.node_mlp(x)
        return P


# TODO Add residuals
class Process(nn.Module):
    def __init__(self, n_layers, mpnn):
        super().__init__()

        self.layers = nn.ModuleList([MPNN(**mpnn) for _ in range(n_layers)])

    def forward(self, P):

        for layer in self.layers:
            residual = P.x
            P = layer(P)

            P.x += residual

        return P


class Decoder(nn.Module):
    def __init__(self, decoder):
        super().__init__()

        self.node_decoder = MLP(**decoder)

    def forward(self, P):
        P.accs = self.node_decoder(P.x)
        return P


class GNS(nn.Module):
    """
    PyTorch Implemenation of a Graph (Neural) Network-based Simulator from 'Learning to simulate complex physics with graph networks'
    """

    def __init__(
        self, k, n_steps, time_step, temperature, encoder, processor, decoder, **kwarg
    ):
        super().__init__()
        self.k = k
        self.n_steps = n_steps
        self.timestep = time_step
        self.temperature = temperature

        self.encoder = Encoder(**encoder)
        self.processor = Process(**processor)
        self.decoder = Decoder(decoder)

    def forward(
        self, coords, x, res_numbers, masses, seq, animation=None, animation_steps=None
    ):

        vels = torch.randn(coords.shape).to(coords.device) * self.temperature
        P = Data(
            x=x, pos=coords, res_numbers=res_numbers, masses=masses, seq=seq, vels=vels
        )

        P.x_native = P.x
        P.randn_coords = P.pos + P.vels * self.n_steps

        for i in range(self.n_steps if not animation_steps else animation_steps):

            P = self._preprocess(P)

            P = self.encoder(P)
            P = self.processor(P)
            P = self.decoder(P)

            P = self._update(P)

            if animation:
                if i % animation == 0:
                    print(
                        f"Saving structure {i//animation} out of {animation_steps//animation}"
                    )
                    save_structure(P.pos, "animation.pdb", P.seq, i // animation)

        coords = P.pos
        return coords

    def _preprocess(self, P):
        """Computing graph connectivity and edge features"""
        k = self.k

        P.n_atoms = n_atoms = P.x.shape[0]
        pos = P.pos
        res_numbers = P.res_numbers

        idx = knn(pos, k + 1)
        senders = idx[:, 0].repeat_interleave(k)
        receivers = idx[:, 1:].reshape(n_atoms * k)

        # Calc Euclidian distance
        diffs = pos[senders] - pos[receivers]
        dists = diffs.norm(dim=1)
        norm_diffs = diffs / dists.clamp(min=0.01).unsqueeze(1)

        # Calc sequence seperation
        seq_sep = abs(res_numbers[senders] - res_numbers[receivers]) / 5
        mask = seq_sep > 1
        seq_sep[mask] = 1

        # Concat edge features
        edges = torch.cat([diffs, dists.unsqueeze(1), seq_sep], dim=1)

        P.edge_attr, P.senders, P.receivers = edges, senders, receivers

        return P

    def _update(self, P):
        """A simple Euler integrator"""

        P.pos = (
            P.pos
            + P.vels * self.timestep
            + 0.5 * P.accs * self.timestep * self.timestep
        )

        P.vels = P.vels + P.accs * self.timestep

        return P


if __name__ == "__main__":

    from dynamics.data.datasets.greener.datamodule import GreenerDataModule
    from dynamics.model.utils.geometric import knn

    dm = GreenerDataModule(
        "/home/cch57/projects/protein_dynamics/dynamics/data/datasets/greener", 1
    )
    val_loader = dm.val_dataloader()

    k = 20

    from hydra import compose, initialize
    from omegaconf import OmegaConf

    initialize(config_path="../../config/model/", job_name="test_app")
    cfg = compose(config_name="gns")

    for P in val_loader:
        print(P)

        P.vels = torch.randn(P.pos.shape) * 0.05

        model = GNS(**cfg.params)
        out = model(P)

        exit()
