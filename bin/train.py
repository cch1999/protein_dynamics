from dynamics.model.dms import DMSWrapper
from dynamics.data.datasets.greener.datamodule import GreenerDataModule
import torch
import pytorch_lightning as pl

#from allostery.model.assemble import prepare_model
#from allostery.data.assemble import prepare_dataloader

import hydra 
from logging import log
from omegaconf import DictConfig, OmegaConf


@hydra.main("../config/", config_name="main")
def train(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    # Log info
    #log.info(f"Running Experiment: {config.name} \n {config.dict()}")

    # Prepare dataloaders and model
    data_module = GreenerDataModule(config.dataset.dir,config.training.batch_size)
    model = DMSWrapper(config)

    # Configure Trainer
    logger = pl.loggers.WandbLogger(log_model=True, project="dynamics")
    trainer = pl.Trainer(logger=logger,
                        max_epochs=config.training.epochs)

    # Train
    trainer.fit(model, data_module)

if __name__ == "__main__":
    train()