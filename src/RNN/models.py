"""
    This script defines the model implementation code and handles the supporting code for training and validation
"""

#TODO: Clean callback function implementation
#TODO: To run on CPU, comment the callback function on_validation_epoch_end()
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from lightning import LightningModule
from numpy.lib.stride_tricks import sliding_window_view
from torchmetrics.regression import (
    R2Score,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
)
from torchmetrics.functional.regression import mean_absolute_error


class EncoderDecoderGRU(LightningModule):

    def __init__(
        self,
        encoder_dim: int = 2,
        decoder_dim: int = 1,
        hidden_size: int = 64,
        encoder_layers: int = 1,
        decoder_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 0.0001,
        normalize: bool = True,
        voltage_min: float = 2.5,
        voltage_max: float = 4.2,
        current_min: float = -2.5,
        current_max: float = 1.0,
    ):
        super().__init__()

        ########## Start : Model Architecture Code ##########
        self.encoder = nn.GRU(
            input_size=encoder_dim,
            hidden_size=hidden_size,
            num_layers=encoder_layers,
            dropout=dropout if encoder_layers > 1 else 0.0,
            batch_first=True,
        )

        self.decoder = nn.GRU(
            input_size=decoder_dim,
            hidden_size=hidden_size,
            num_layers=decoder_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.output_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, decoder_dim),
        )
        ########## End : Model Architecture Code ##########

        self.loss = nn.MSELoss()

        self.train_r2_score, self.val_r2_score, self.test_r2_score = (
            R2Score(),
            R2Score(),
            R2Score(),
        )
        self.train_mae, self.val_mae, self.test_mae = (
            MeanAbsoluteError(),
            MeanAbsoluteError(),
            MeanAbsoluteError(),
        )
        self.train_mape, self.val_mape, self.test_mape = (
            MeanAbsolutePercentageError(),
            MeanAbsolutePercentageError(),
            MeanAbsolutePercentageError(),
        )
        self.train_rmse, self.val_rmse, self.test_rmse = (
            MeanSquaredError(squared=False),
            MeanSquaredError(squared=False),
            MeanSquaredError(squared=False),
        )
        self.decoder_layers = decoder_layers
        self.lr = lr
        self.normalize = normalize
        self.voltage_min = voltage_min
        self.voltage_max = voltage_max
        self.current_min = current_min
        self.current_max = current_max
        self.save_hyperparameters()

    def forward(self, encoder_input, decoder_input):
        _, hidden_state = self.encoder(encoder_input)
        hidden_state = hidden_state.expand(self.decoder_layers, -1, -1).contiguous()
        output, _ = self.decoder(decoder_input, hidden_state)
        output = self.output_layer(output)
        return output

    def training_step(self, batch):
        encoder_input, decoder_input, decoder_output = batch

        predicted_output = self(encoder_input, decoder_input)
        train_loss = self.loss(predicted_output, decoder_output)
        # Denormalize the output to calculate accuracy metrics
        if self.normalize:
            decoder_output = (
                decoder_output * (self.voltage_max - self.voltage_min)
                + self.voltage_min
            )
            predicted_output = (
                predicted_output * (self.voltage_max - self.voltage_min)
                + self.voltage_min
            )
        # Calculate and log metrics
        self.train_mae(predicted_output, decoder_output)
        self.train_mape(predicted_output, decoder_output)
        self.train_rmse(predicted_output, decoder_output)
        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )
        self.log(
            "train_mae", self.train_mae, on_epoch=True, on_step=True, prog_bar=True
        )
        self.log("train_mape", self.train_mape, on_epoch=True, on_step=True)
        self.log("train_rmse", self.train_rmse, on_epoch=True, on_step=True)
        return train_loss

    def validation_step(self, batch):
        encoder_input, decoder_input, decoder_output = batch

        predicted_output = self(encoder_input, decoder_input)
        val_loss = self.loss(predicted_output, decoder_output)
        # Denormalize the output to calculate accuracy metrics
        if self.normalize:
            decoder_output = (
                decoder_output * (self.voltage_max - self.voltage_min)
                + self.voltage_min
            )
            predicted_output = (
                predicted_output * (self.voltage_max - self.voltage_min)
                + self.voltage_min
            )
        # Calculate and log metrics
        self.val_mae(predicted_output, decoder_output)
        self.val_mape(predicted_output, decoder_output)
        self.val_rmse(predicted_output, decoder_output)
        self.log(
            "val_loss",
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )
        self.log("val_mae", self.val_mae, on_epoch=True, on_step=True, prog_bar=True)
        self.log("val_mape", self.val_mape, on_epoch=True, on_step=True)
        self.log("val_rmse", self.val_rmse, on_epoch=True, on_step=True)
        return val_loss

    def on_validation_epoch_end(
        self, encoder_input_length=20, decoder_input_length=1980
    ):
        """
        This function is called at the end of every epoch.
        It computes the autoregressive loss by using only a small window of initial ground truth measurements.
        The first stage uses ground truth for the prediction, while the later stages use previous predicted data points for future predictions
        """
        #TODO: Fix this quick hard coded implementation
        curve_dir = {}
        curve_dir[1] = (
            "/home/mazin/Projects/Thesis/RNN/Data/149/1/1_4_0_0.csv"
        )
        curve_dir[2] = (
            "/home/mazin/Projects/Thesis/RNN/Data/149/2/2_4_0_0.csv"
        )
        curve_dir[3] = (
            "/home/mazin/Projects/Thesis/RNN/Data/149/3/3_4_0_0.csv"
        )
        curve_dir[4] = (
            "/home/mazin/Projects/Thesis/RNN/Data/149/4/4_4_0_0.csv"
        )
        curve_dir[5] = (
            "/home/mazin/Projects/Thesis/RNN/Data/149/5/5_4_0_0.csv"
        )
        curve_dir[6] = (
            "/home/mazin/Projects/Thesis/RNN/Data/149/6/6_4_0_0.csv"
        )

        for profile in curve_dir:
            curve = pd.read_csv(curve_dir[profile], usecols=["voltage", "current"])
            # Calculate the residual length for the final sample
            last_sample_length = (
                curve[encoder_input_length:].shape[0] % decoder_input_length
            )
            # Apply MinMax Normalization -> [0,1]
            curve["voltage"] = (curve["voltage"] - self.voltage_min) / (
                self.voltage_max - self.voltage_min
            )
            curve["current"] = (curve["current"] - self.current_min) / (
                self.current_max - self.current_min
            )
            actual_current = torch.tensor(
                curve["current"], dtype=torch.float
            )
            actual_voltage = torch.tensor(curve["voltage"], dtype=torch.float)
            predicted_voltage = torch.zeros(curve.shape[0], dtype=torch.float)
            curve_sliding_samples = sliding_window_view(
                curve[encoder_input_length:], (decoder_input_length, 2)
            ).squeeze()
            curve_samples = torch.tensor(
                np.concatenate(
                    (
                        curve_sliding_samples[::decoder_input_length],
                        curve_sliding_samples[-1:],
                    ),
                    axis=0,
                ),
                dtype=torch.float,
            )
            encoder_input = (
                torch.tensor(curve[:encoder_input_length].to_numpy(), dtype=torch.float)
                .unsqueeze(0)
                .to("cuda")
            )

            decoder_input = curve_samples[0, :, 1].reshape(1, -1, 1).to("cuda")
            predicted_voltage[:encoder_input_length] = encoder_input[:, :, 0]
            self.eval()
            with torch.inference_mode():
                output = self(encoder_input, decoder_input)
            predicted_voltage[
                encoder_input_length : encoder_input_length + decoder_input_length
            ] = output.squeeze()
            for i in range(1, curve_samples.shape[0]):
                if i != curve_samples.shape[0] - 1:
                    encoder_input = torch.cat(
                        [
                            output[:, -encoder_input_length:],
                            decoder_input[:, -encoder_input_length:],
                        ],
                        dim=-1,
                    )
                    decoder_input = curve_samples[i, :, 1].reshape(1, -1, 1).to("cuda")
                    self.eval()
                    with torch.inference_mode():
                        output = self(encoder_input, decoder_input)
                    predicted_voltage[
                        encoder_input_length
                        + i * decoder_input_length : encoder_input_length
                        + (i + 1) * decoder_input_length
                    ] = output.squeeze()
                else:
                    encoder_input = torch.cat(
                        [
                            output[
                                :,
                                last_sample_length
                                - decoder_input_length
                                - encoder_input_length : last_sample_length
                                - decoder_input_length,
                            ],
                            decoder_input[
                                :,
                                last_sample_length
                                - decoder_input_length
                                - encoder_input_length : last_sample_length
                                - decoder_input_length,
                            ],
                        ],
                        dim=-1,
                    )

                    decoder_input = curve_samples[-1, :, 1].reshape(1, -1, 1).to("cuda")
                    self.eval()
                    with torch.inference_mode():
                        output = self(encoder_input, decoder_input)
                    predicted_voltage[-decoder_input_length:] = output.squeeze()
                    # Denormalize
                    actual_current = (
                        actual_current * (self.current_max - self.current_min)
                        + self.current_min
                    )
                    predicted_voltage = (
                        predicted_voltage * (self.voltage_max - self.voltage_min)
                        + self.voltage_min
                    )
                    actual_voltage = (
                        actual_voltage * (self.voltage_max - self.voltage_min)
                        + self.voltage_min
                    )
            self.log(
                f"profile_{profile}_mae",
                mean_absolute_error(predicted_voltage, actual_voltage),
                on_epoch=True,
            )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
