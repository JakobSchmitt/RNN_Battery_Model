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
import time

from RNN.datamodules import UniDataModule
from RNN.models import EncoderDecoderGRU

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

import torch
torch.set_float32_matmul_precision('high')

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
        normalize=cfg.method.normalize,
        voltage_min=cfg.method.voltage_min,
        voltage_max=cfg.method.voltage_max,
        current_min=cfg.method.current_min,
        current_max=cfg.method.current_max,
    )

    if cfg.model_checkpoint == True:
        pretrained_weights_path = Path(cfg.pretrained_weights_dir) / 'weights' / 'best_model' 
        model = EncoderDecoderGRU.load_from_checkpoint(list(pretrained_weights_path.glob('*.ckpt'))[0])
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

    all_checkpoint_callback = ModelCheckpoint(
        dirpath="weights/all_epochs/",
        filename=wandb_logger.experiment.name
        + "-{epoch:02d}-{train_loss:.7f}-{val_loss:.7f}-{combined_val_loss:.7f}",
        save_top_k=-1,#cfg.save_top_k,
        every_n_epochs=1,
        # monitor="combined_val_loss",#"val_loss",
        monitor=None,  # Do not monitor any metric; this disables stateful tracking, else error due two stateful callbacks!

    )
    best_checkpoint_callback = ModelCheckpoint(
        dirpath="weights/best_model/",
        filename=wandb_logger.experiment.name
        + "-{epoch:02d}-{train_loss:.7f}-{val_loss:.7f}-{combined_val_loss:.7f}-best",
        save_top_k=1,
        monitor="combined_val_loss",#"val_loss",
        
    )
    
    load_best_weights_callback = LoadBestWeightsCallback(checkpoint_callback=best_checkpoint_callback)


    trainer = L.Trainer(
        callbacks=[all_checkpoint_callback, best_checkpoint_callback, load_best_weights_callback],
        max_epochs=cfg.epochs,
        accelerator=cfg.device,
        logger=wandb_logger,
    )

    logging.info(f"{cfg.method.mode} : Starting training ...")
    
    trainer.fit(model, data)
    

class LoadBestWeightsCallback(Callback):
    def __init__(self, checkpoint_callback):
        """
        Args:
            checkpoint_callback: The ModelCheckpoint instance used during training.
        """
        super().__init__()
        self.checkpoint_callback = checkpoint_callback

    def on_train_epoch_start(self, trainer, pl_module):
        """
        Called at the start of each training epoch.
        """
        # Get the path to the best checkpoint
        best_model_path = self.checkpoint_callback.best_model_path

        if best_model_path:
            # Load state dictionary from the checkpoint file
            state_dict = torch.load(best_model_path)["state_dict"]
            pl_module.load_state_dict(state_dict)
            logging.info(f"Loaded best weights from {best_model_path}")
        else:
            logging.warning("No best model checkpoint available to load.")


if __name__ == "__main__":
    main()
