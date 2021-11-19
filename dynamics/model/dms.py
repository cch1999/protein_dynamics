import torch
import pytorch_lightning as pl


class DMSWrapper(pl.LightningModule):
    def __init__(self, config):
        super(DMSWrapper).__init__()

        if config.model.name == "PBMP":
            self.model = PBMP(config)

        # Set loss function
        if config.loss.name == 'rmds':
            self.loss_fn = RMSD

    def forward(self, P):

        # Init random velocities
        P.vels = torch.randn(P.coords.shape) * self.config.simulator.temperature
		P.accs_last = torch.zeros(P.coords.shape)
		#P.randn_coords = coords + vels * timestep * n_steps

        return self.model(P

    def training_step(self, P):
        
        pred = self.forward(P)
        loss = self.loss_fn(pred, y)
        
        self.log("train_loss", loss)
        return loss

    def validation_step(self):
		        
        pred = self.forward(P)
        loss = self.loss_fn(pred, y)
        
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer