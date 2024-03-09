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
        + "-{epoch:02d}-{train_loss:.6f}-{val_loss:.6f}-{val_mae:.3f}",
        monitor="val_loss",
        # every_n_train_steps=cfg.every_n_train_steps,
        mode="min",
        save_top_k=10,
    )

    checkpoint_callback_train = ModelCheckpoint(
        dirpath="weights/",
        filename=cfg.method.mode
        + "-monitor_train"
        + "-{epoch:02d}-{train_loss:.6f}-{val_loss:6f}-{val_mae:.3f}",
        monitor="train_loss",
        # every_n_train_steps=cfg.every_n_train_steps,
        mode="min",
        save_top_k=10,
    )
    wandb_logger = WandbLogger(**cfg.logger)

    trainer = L.Trainer(
        callbacks=[checkpoint_callback_val, checkpoint_callback_train],
        max_epochs=cfg.epochs,
        # log_every_n_steps=cfg.log_every_n_steps,
        # val_check_interval=cfg.val_check_interval,
        accelerator="gpu",
        devices=cfg.num_gpus,
        strategy=cfg.strategy,
        logger=wandb_logger,
    )

    logging.info(f"{cfg.method.mode} : Starting training ...")
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
