from dynamics.model.dms import DMSWrapper
from dynamics.data.datasets.greener.datamodule import GreenerDataModule
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

    # Configure Trainer
    logger = pl.loggers.WandbLogger(log_model='all', project="dynamics", config=config)
    logger.watch(model)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=config.training.epochs,
        log_every_n_steps=config.training.logging_freq,
        flush_logs_every_n_steps=config.training.logging_freq,
        val_check_interval=0.001,
    )

    # Train
    trainer.fit(model, data_module)


if __name__ == "__main__":
    train()
