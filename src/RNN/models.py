"""
    This script defines the model implementation code and handles the supporting code for training and validation
"""

#TODO: Clean callback function implementation
#TODO: To run on CPU, comment the callback function on_validation_epoch_end()
import torch
import numpy as np
import time
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

torch.set_float32_matmul_precision('high')

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
        
        # Load curves into memory during initialization
        self.curve_dir = {}
        # Todo: change hard coded paths
        self.curve_dir[1] = (
             r"C:\Users\s8940173\Dokumente\GitHub\RNN_Battery_Model\Data/149/1/1_4_0_0.csv"
         )
        self.curve_dir[2] = (
             r"C:\Users\s8940173\Dokumente\GitHub\RNN_Battery_Model\Data/149/2/2_4_0_0.csv"
         )
        self.curve_dir[3] = (
             r"C:\Users\s8940173\Dokumente\GitHub\RNN_Battery_Model\Data/149/3/3_4_0_0.csv"
         )
        self.curve_dir[4] = (
             r"C:\Users\s8940173\Dokumente\GitHub\RNN_Battery_Model\Data/149/4/4_4_0_0.csv"
         )
        self.curve_dir[5] = (
             r"C:\Users\s8940173\Dokumente\GitHub\RNN_Battery_Model\Data/149/5/5_4_0_0.csv"
         )
        self.curve_dir[6] = (
             r"C:\Users\s8940173\Dokumente\GitHub\RNN_Battery_Model\Data/149/6/6_4_0_0.csv"
         )
        self.curves = self.load_curves()
        
        
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
        self.log("train_loss",train_loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log("train_mae", self.train_mae, on_epoch=True, on_step=True, prog_bar=True)
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
        self.log("val_loss",val_loss, on_epoch=True,  on_step=True, prog_bar=False)
        self.log("val_mae", self.val_mae, on_epoch=True, on_step=True, prog_bar=False)
        self.log("val_mape", self.val_mape, on_epoch=True, on_step=True)
        self.log("val_rmse", self.val_rmse, on_epoch=True, on_step=True)

        return val_loss

    def on_validation_epoch_end(self, encoder_input_length=20, decoder_input_length=1980):
        """
        This function is called at the end of every epoch.
        It computes the autoregressive loss by using only a small window of initial ground truth measurements.
        The first stage uses ground truth for the prediction, while the later stages use previous predicted data points for future predictions
        """
        
        # TODO: could be parallelised, so that all 6 profiles are predicted simulaneously?!
        start_time = time.time()

        profile_losses = []

        for profile, curve in self.curves.items():

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
            # use first 1980 current values as decoder input
            decoder_input = curve_samples[0, :, 1].reshape(1, -1, 1).to("cuda")
            # start prediction array with inital 20 voltage values 
            predicted_voltage[:encoder_input_length] = encoder_input[:, :, 0]
            # necessary to call eval() for prediction mode of model
            self.eval()
            with torch.inference_mode():
                output = self(encoder_input, decoder_input)
            # store output sequence of length 1980 in prediction voltage array (len of initial discharge curve)
            predicted_voltage[
                encoder_input_length : encoder_input_length + decoder_input_length
            ] = output.squeeze()
            for i in range(1, curve_samples.shape[0]):
                # last prediction segment for profile
                if i != curve_samples.shape[0] - 1:
                    # setup current encoder_input : consisting of last 20 predictions from prev output 
                    encoder_input = torch.cat(
                        [
                            output[:, -encoder_input_length:],
                            decoder_input[:, -encoder_input_length:],
                        ],
                        dim=-1,
                    )
                    # get next current input 
                    decoder_input = curve_samples[i, :, 1].reshape(1, -1, 1).to("cuda")
                    self.eval()
                    with torch.inference_mode():
                        output = self(encoder_input, decoder_input)
                    # store prediction
                    predicted_voltage[
                        encoder_input_length
                        + i * decoder_input_length : encoder_input_length
                        + (i + 1) * decoder_input_length
                    ] = output.squeeze()
                else:
                    # setup this way, as encoder doesn't learn to initalise low voltage levels - lowest voltage level is length - 2000 ! so simply initalising with last_sample_length-20 doesnt work!
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
            
            profile_loss = mean_absolute_error(predicted_voltage, actual_voltage)
            profile_losses.append(profile_loss)


            self.log(
                f"profile_{profile}_mae",
                mean_absolute_error(predicted_voltage, actual_voltage),
                on_epoch=True,
            )
        
        # Compute combined validation loss
        avg_profile_loss = torch.stack(profile_losses).mean()
        val_loss = self.trainer.callback_metrics["val_loss"]
        combined_val_loss = val_loss + avg_profile_loss
        # Log combined loss
        self.log("combined_val_loss", combined_val_loss, on_epoch=True, prog_bar=False)
        
        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.log("on_validation_epoch_end_time", elapsed_time, prog_bar=False)
        try:
            time_consumption = self.trainer.callback_metrics["on_validation_epoch_end_time"]
            print(f"Validation profile time consumtion: {time_consumption*100:.1f} s")
        except:
            pass

        # call for further behavourial metric computation : OCV variance minimisation
        # self.on_validation_epoch_end2()
        
    
    def on_validation_epoch_end2(self,encoder_input_length=20, decoder_input_length=1980):
        # what to minimise ? the spread between profile based OCV approx or random current or segment shuffle ?
        pass
        
    
    def test_model(self, encoder_input_length=20, decoder_input_length=1980):
        """
        Evaluate the model on test curves using autoregressive prediction.
        This is similar to the on_validation_epoch_end method but adapted for testing.
        Args:
            curves: A dictionary containing test curves (key: profile name, value: curve data).
            encoder_input_length: Number of initial data points used for the encoder input.
            decoder_input_length: Length of each predicted sequence segment.
        """

    
        # TODO: could be parallelised, so that all 6 profiles are predicted simulaneously?!
        start_time = time.time()

        profile_losses = []
        current_profile_dict,voltage_profile_dict,prediction_profile_dict = dict(),dict(),dict()


        for profile, curve in self.curves.items():

            # Calculate the residual length for the final sample
            last_sample_length = (
                curve[encoder_input_length:].shape[0] % decoder_input_length
            )
            # Apply MinMax Normalization -> [0,1]
            curve["voltage"] = (curve["voltage"] - self.voltage_min) / ( self.voltage_max - self.voltage_min)
            curve["current"] = (curve["current"] - self.current_min) / (self.current_max - self.current_min)
            actual_current = torch.tensor(curve["current"], dtype=torch.float)
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
            # use first 1980 current values as decoder input
            decoder_input = curve_samples[0, :, 1].reshape(1, -1, 1).to("cuda")
            # start prediction array with inital 20 voltage values 
            predicted_voltage[:encoder_input_length] = encoder_input[:, :, 0]
            # necessary to call eval() for prediction mode of model
            self.eval()
            with torch.inference_mode():
                output = self(encoder_input, decoder_input)
            # store output sequence of length 1980 in prediction voltage array (len of initial discharge curve)
            predicted_voltage[
                encoder_input_length : encoder_input_length + decoder_input_length
            ] = output.squeeze()
            for i in range(1, curve_samples.shape[0]):
                # last prediction segment for profile
                if i != curve_samples.shape[0] - 1:
                    # setup current encoder_input : consisting of last 20 predictions from prev output 
                    encoder_input = torch.cat(
                        [
                            output[:, -encoder_input_length:],
                            decoder_input[:, -encoder_input_length:],
                        ],
                        dim=-1,
                    )
                    # get next current input 
                    decoder_input = curve_samples[i, :, 1].reshape(1, -1, 1).to("cuda")
                    self.eval()
                    with torch.inference_mode():
                        output = self(encoder_input, decoder_input)
                    # store prediction
                    predicted_voltage[
                        encoder_input_length
                        + i * decoder_input_length : encoder_input_length
                        + (i + 1) * decoder_input_length
                    ] = output.squeeze()
                else:
                    # setup this way, as encoder doesn't learn to initalise low voltage levels - lowest voltage level is length - 2000 ! so simply initalising with last_sample_length-20 doesnt work!
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
            
            profile_loss = mean_absolute_error(predicted_voltage, actual_voltage)
            profile_losses.append(profile_loss)
            current_profile_dict[profile] = actual_current
            voltage_profile_dict[profile] = actual_voltage
            prediction_profile_dict[profile] = predicted_voltage
        
        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
    
        return current_profile_dict, voltage_profile_dict, prediction_profile_dict, profile_losses
   
    
        
        

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def load_curves(self):
       """Load all curve CSV files into memory."""
       curves = {}
       for key, path in self.curve_dir.items():
           curves[key] = pd.read_csv(path, usecols=["voltage", "current"])
       return curves
