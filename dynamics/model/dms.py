import torch
import pytorch_lightning as pl

from copy import deepcopy

from dynamics.model.pbmp import PBMP
from dynamics.model.gns import GNS
from dynamics.model.egns import EGNS
from dynamics.utils.loss import rmsd, msd


class DMSWrapper(pl.LightningModule):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.save_hyperparameters()

		if config.model.name == "pbmp":
			self.model = PBMP(**config.model.params)
		if config.model.name == "gns":
			self.model = GNS(**config.model.params)
		if config.model.name == "egns":
			self.model = EGNS(**config.model.params)

		# Set loss function
		if config.training.loss == "rmsd":
			self.loss_fn = rmsd
		elif config.training.loss == "msd":
			self.loss_fn = msd

	def forward(self, coords, x, res_numbers, masses, seq, animation=None, animation_steps=None):
		if self.config.model.name == "pbmp":
			return self.model(coords, x, res_numbers, masses, seq, animation=animation, animation_steps=animation_steps)

	def training_step(self, P, batch_idx):

		coords = deepcopy(P.native_coords)
		x, res_numbers, masses, seq = P.x, P.res_numbers, P.masses, P.seq

		coords_out = self.forward(coords, x, res_numbers, masses, seq)

		loss, passed = self.loss_fn(coords_out, P.native_coords)
		#basic_loss, _ = self.loss_fn(P.randn_coords, P.native_coords)

		self.log("train_loss", loss)
		#self.log("train_corrected_loss", basic_loss - loss)
		return loss


	def validation_step(self, P, batch_idx):

		coords = deepcopy(P.native_coords)
		x, res_numbers, masses, seq = P.x, P.res_numbers, P.masses, P.seq

		coords_out = self.forward(coords, x, res_numbers, masses, seq)

		loss, passed = self.loss_fn(coords_out, P.native_coords)
		#basic_loss, _ = self.loss_fn(P.randn_coords, P.native_coords)

		self.log("val_loss", loss)
		#self.log("val_corrected_loss", basic_loss - loss)

		return loss

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
		return optimizer
