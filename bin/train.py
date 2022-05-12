from html import entities
from dynamics.model.dms import DMSWrapper
from dynamics.data.datasets.greener.datamodule import GreenerDataModule
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.profiler import AdvancedProfiler

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
    logger = pl.loggers.WandbLogger(log_model='all', project="dynamics", entity='cch1999', name=config.name, config=config)
    logger.watch(model)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

    trainer = pl.Trainer(
        logger=logger if config.name != 'test' else None,
        callbacks=[checkpoint_callback,
                    ModelSummary(max_depth=3)],
        max_epochs=config.training.epochs,
        log_every_n_steps=config.training.logging_freq,
        flush_logs_every_n_steps=config.training.logging_freq,
        val_check_interval=config.training.val_check_interval,
        profiler=AdvancedProfiler(dirpath='outputs', filename='report.txt'),
        fast_dev_run=1,
    )

    # Train
    trainer.fit(model, data_module)


if __name__ == "__main__":
    train()
