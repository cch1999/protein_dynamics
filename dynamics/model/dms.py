import torch
import pytorch_lightning as pl

from dynamics.model.pbmp import PBMP
from dynamics.utils.loss import rmsd


class DMSWrapper(pl.LightningModule):
	def __init__(self, config):
		super().__init__()
		self.config = config

		if config.model.name == "pbmp":
			self.model = PBMP(**config.model.params)

		# Set loss function
		if config.training.loss == 'rmsd':
			self.loss_fn = rmsd

	def forward(self, P):

		# Init random velocities
		P.vels = torch.randn(P.pos.shape) * self.config.simulator.temperature
		P.accs_last = torch.zeros(P.pos.shape)
		#P.randn_coords = coords + vels * timestep * n_steps

		return self.model(P)

	def training_step(self, P, batch_idx):
		
		P = self.forward(P)
		loss, passed = self.loss_fn(P.coords, P.native_coords)
		
		self.log("train_loss", loss)
		return loss

	def validation_step(self, P, batch_idx):
				
		P = self.forward(P)
		loss = self.loss_fn(P.coords, P.native_coords)
		
		self.log("val_loss", loss)
		return loss

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer