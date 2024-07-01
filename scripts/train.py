"""

train.py

This script trains the model. It is designed to run from the command line and can take various arguments to update the model, data and training process.
The scripts takes the default arguments  as input mentioned in the config:
- train.yaml
- method/rnn.yaml

Usage:
    ex: python scripts/train.py
    # To update the parameters, either change the config files or pass the arguments from the command line
    ex: python scripts/train.py epochs=10 method.stride=35

"""

import hydra
import logging
from pathlib import Path

from RNN.datamodules import UniDataModule
from RNN.models import EncoderDecoderGRU

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint


@hydra.main(version_base="1.1", config_path="../config/", config_name="train")
def main(cfg):

    # data_dir = Path(hydra.utils.to_absolute_path(cfg.method.root_dir))

    logging.info(f"{cfg.method.mode} : Loading data ...")
    data = UniDataModule(
        mode=cfg.method.mode,
        root_dir=Path(hydra.utils.to_absolute_path(cfg.method.root_dir)),
        encoder_input_length=cfg.method.encoder_input_length,
        stride=cfg.method.stride,
        decoder_input_length=cfg.method.decoder_input_length,
        batch_size=cfg.method.batch_size,
        num_workers=cfg.num_workers,
        shuffle=cfg.method.shuffle,
        normalize=cfg.method.normalize,
        voltage_min=cfg.method.voltage_min,
        voltage_max=cfg.method.voltage_max,
        current_min=cfg.method.current_min,
        current_max=cfg.method.current_max,
    )

    if cfg.model_checkpoint == True:
        model = EncoderDecoderGRU.load_from_checkpoint(cfg.pretrained_weights)
        logging.info(f"{cfg.method.mode} : Model loaded using checkpoint")
    else:
        model = EncoderDecoderGRU(
            encoder_dim=cfg.method.encoder_dim,
            decoder_dim=cfg.method.decoder_dim,
            hidden_size=cfg.method.hidden_size,
            encoder_layers=cfg.method.encoder_layers,
            decoder_layers=cfg.method.decoder_layers,
            dropout=cfg.method.dropout,
            lr=cfg.lr,
            normalize=cfg.method.normalize,
            voltage_min=cfg.method.voltage_min,
            voltage_max=cfg.method.voltage_max,
            current_min=cfg.method.current_min,
            current_max=cfg.method.current_max,
        )
        logging.info(f"{cfg.method.mode} : Model loaded without checkpoint")

    # Initialize wandb
    wandb_logger = WandbLogger(**cfg.logger)
    wandb_logger.experiment.config["model_checkpoint"] = cfg.model_checkpoint
    if cfg.model_checkpoint == True:
        wandb_logger.experiment.config["pretrained_weights"] = cfg.pretrained_weights

    checkpoint_callback = ModelCheckpoint(
        dirpath="weights/",
        filename=wandb_logger.experiment.name
        + "-{epoch:02d}-{train_loss:.7f}-{val_loss:.7f}-{val_mae:.5f}",
        save_top_k=cfg.save_top_k,
        monitor="val_loss",
    )

    trainer = L.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=cfg.epochs,
        accelerator=cfg.device,
        logger=wandb_logger,
    )

    logging.info(f"{cfg.method.mode} : Starting training ...")
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
