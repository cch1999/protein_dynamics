import torch
import pytorch_lightning as pl

from dynamics.model.pbmp import PBMP
from dynamics.model.gns import GNS
from dynamics.utils.loss import rmsd


class DMSWrapper(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.model.name == "pbmp":
            self.model = PBMP(**config.model.params)
        if config.model.name == "gns":
            self.model = GNS(**config.model.params)

        # Set loss function
        if config.training.loss == "rmsd":
            self.loss_fn = rmsd

    def forward(self, P):

        # Init random velocities
        P.vels = torch.randn(P.pos.shape) * self.config.simulator.temperature
        P.accs_last = torch.zeros(P.pos.shape)

        return self.model(P)

    def training_step(self, P, batch_idx):

        P = self.forward(P)
        loss, _ = self.loss_fn(P.coords, P.native_coords)
        basic_loss, _ = self.loss_fn(P.randn_coords, P.native_coords)

        self.log("train_loss", loss)
        self.log("train_corrected_loss", basic_loss - loss)
        return loss

    def validation_step(self, P, batch_idx):

        P = self.forward(P)
        loss, _ = self.loss_fn(P.coords, P.native_coords)
        basic_loss, _ = self.loss_fn(P.randn_coords, P.native_coords)

        self.log("val_loss", loss)
        self.log("train_corrected_loss", basic_loss - loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        return optimizer
