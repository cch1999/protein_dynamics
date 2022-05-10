from dynamics.model.dms import DMSWrapper
from dynamics.data.datasets.greener.datamodule import GreenerDataModule
from dynamics.utils.loss import rmsd
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint

import hydra
from logging import log
from omegaconf import DictConfig, OmegaConf


@hydra.main("../config/", config_name="main")
def train(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    # Prepare dataloaders and model
    data_module = GreenerDataModule(config.dataset.dir, config.training.batch_size, config.dataset.fraction)
    model = DMSWrapper(config)

    # Train
    #trainer.fit(model, data_module)

    path = '/home/cch57/projects/protein_dynamics/outputs/2022-03-03/10-57-29/dynamics/pje4fvkr/checkpoints/epoch=1-step=2603.ckpt'

    model = model.load_from_checkpoint(path)

    print(model.config)
    # prints the learning_rate you used in this checkpoint

    x = next(iter(data_module.val_dataloader()))

    model.eval()
    out = model.forward(x, 1, 100)

    print(out)
    print('RMSD: ', rmsd(x.coords, out.coords))




if __name__ == "__main__":
    train()
