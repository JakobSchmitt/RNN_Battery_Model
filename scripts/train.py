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

    logging.info(f"{cfg.method.mode} : Loading data ...")
    data = UniDataModule(
        mode=cfg.method.mode,
        root_dir=Path(cfg.method.root_dir),
        encoder_input_length=cfg.method.encoder_input_length,
        stride=cfg.method.stride,
        decoder_input_length=cfg.method.decoder_input_length,
        batch_size=cfg.method.batch_size,
        num_workers=cfg.num_workers,
        shuffle=cfg.method.shuffle,
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
        )
        logging.info(f"{cfg.method.mode} : Model loaded without checkpoint")

    checkpoint_callback_val = ModelCheckpoint(
        dirpath="weights/",
        filename=cfg.method.mode
        + "-monitor_val"
        + "-{epoch:02d}-{train_loss:.7f}-{val_loss:.7f}-{val_mae:.5f}",
        monitor="val_loss",
        mode="min",
        save_top_k=-1,
    )

    wandb_logger = WandbLogger(**cfg.logger)
    wandb_logger.experiment.config["test"] = cfg.test
    wandb_logger.experiment.config["model_checkpoint"] = cfg.model_checkpoint
    if cfg.model_checkpoint == True:
        wandb_logger.experiment.config["pretrained_weights"] = cfg.pretrained_weights

    trainer = L.Trainer(
        callbacks=[checkpoint_callback_val],
        max_epochs=cfg.epochs,
        accelerator="gpu",
        devices=cfg.num_gpus,
        strategy=cfg.strategy,
        logger=wandb_logger,
    )

    if not cfg.test:
        logging.info(f"{cfg.method.mode} : Starting training ...")
        trainer.fit(model, data)
    else:
        logging.info(f"{cfg.method.mode} : Starting validating ...")
        trainer.validate(model, data)
        logging.info(f"{cfg.method.mode} : Starting testing ...")
        trainer.test(model, data)


if __name__ == "__main__":
    main()
