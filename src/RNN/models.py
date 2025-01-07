"""
    This script defines the model implementation code and handles the supporting code for training and validation
"""

#TODO: Clean callback function implementation
#TODO: To run on CPU, comment the callback function on_validation_epoch_end()
import torch
import numpy as np
from copy import deepcopy
import time
import os
from pathlib import Path
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
import alphashape
from shapely.geometry import Polygon, MultiPolygon

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
        encoder_input_length = 20,
        decoder_input_length= 1980,
        gpu = True,
        relax_loss = False,
        pred_data_root_dir=None,#Path('C:/Users/s8940173/Dokumente/GitHub/RNN_Battery_Model/Data/121_stoch'),
        pred_data_dirs=None,#['0', '2', '5', '7', '8', '9', '10', '11', '12', '1', '4', '6', '13', '3']
        val_on_epoch_end=True
        
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
        self.encoder_input_length = encoder_input_length
        self.decoder_input_length =decoder_input_length
        self.relax_loss = relax_loss
        self.gpu = gpu
        self.area_loss_check = False # init for area loss computation decision
        if self.gpu:
            self.comp_mode = "cuda" # GPU computation
        else:
            self.comp_mode = "cpu" # CPU computation

        
        # Load curves into memory during initialization
        self.curve_dir = dict(zip(pred_data_dirs,[pred_data_root_dir / dir_id for dir_id in pred_data_dirs]))      
        self.curves = self.load_curves(self.curve_dir)
        
        self.val_on_epoch_end =  val_on_epoch_end

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
        self.log("val_loss",val_loss, on_epoch=True,  on_step=True, prog_bar=True)
        self.log("val_mae", self.val_mae, on_epoch=True, on_step=True, prog_bar=True)
        self.log("val_mape", self.val_mape, on_epoch=True, on_step=True, prog_bar=False)
        self.log("val_rmse", self.val_rmse, on_epoch=True, on_step=True, prog_bar=False)


        return val_loss

    def on_validation_epoch_end(self):
        """
        This function is called at the end of every epoch.
        It computes the autoregressive loss by using only a small window of initial ground truth measurements.
        The first stage uses ground truth for the prediction, while the later stages use previous predicted data points for future predictions
        """
        if self.val_on_epoch_end:

            # TODO: could be parallelised, so that all 6 profiles are predicted simulaneously?!
            # start_time = time.time()
            
            profile_losses = []
    
            for profile, curve in deepcopy(self.curves).items():
    
                # Calculate the residual length for the final sample
                last_sample_length = (
                    curve[self.encoder_input_length:].shape[0] % self.decoder_input_length
                )
                # Apply MinMax Normalization -> [0,1]
                curve["voltage"] = (curve["voltage"] - self.voltage_min) / (self.voltage_max - self.voltage_min)
                curve["current"] = (curve["current"] - self.current_min) / (self.current_max - self.current_min)
    
                actual_voltage = torch.tensor(curve["voltage"], dtype=torch.float)
                predicted_voltage = torch.zeros(curve.shape[0], dtype=torch.float)
                curve_sliding_samples = sliding_window_view(
                    curve[self.encoder_input_length:], (self.decoder_input_length, 2)
                ).squeeze()
                curve_samples = torch.tensor(
                    np.concatenate(
                        (
                            curve_sliding_samples[::self.decoder_input_length],
                            curve_sliding_samples[-1:],
                        ),
                        axis=0,
                    ),
                    dtype=torch.float,
                )
                encoder_input = (
                    torch.tensor(curve[:self.encoder_input_length].to_numpy(), dtype=torch.float)
                    .unsqueeze(0)
                    .to(self.comp_mode)
                )
                # use first 1980 current values as decoder input
                decoder_input = curve_samples[0, :, 1].reshape(1, -1, 1).to(self.comp_mode)
                # start prediction array with inital 20 voltage values 
                predicted_voltage[:self.encoder_input_length] = encoder_input[:, :, 0]
                # necessary to call eval() for prediction mode of model
                self.eval()
                with torch.inference_mode():
                    output = self(encoder_input, decoder_input)
                # store output sequence of length 1980 in prediction voltage array (len of initial discharge curve)
                predicted_voltage[
                    self.encoder_input_length : self.encoder_input_length + self.decoder_input_length
                ] = output.squeeze()
                for i in range(1, curve_samples.shape[0]):
                    # last prediction segment for profile
                    if i != curve_samples.shape[0] - 1:
                        # setup current encoder_input : consisting of last 20 predictions from prev output 
                        encoder_input = torch.cat(
                            [
                                output[:, -self.encoder_input_length:],
                                decoder_input[:, -self.encoder_input_length:],
                            ],
                            dim=-1,
                        )
                        # get next current input 
                        decoder_input = curve_samples[i, :, 1].reshape(1, -1, 1).to(self.comp_mode)
                        self.eval()
                        with torch.inference_mode():
                            output = self(encoder_input, decoder_input)
                        # store prediction
                        predicted_voltage[
                            self.encoder_input_length
                            + i * self.decoder_input_length : self.encoder_input_length
                            + (i + 1) * self.decoder_input_length
                        ] = output.squeeze()
                    elif last_sample_length>0:
                        # setup this way, as encoder doesn't learn to initalise low voltage levels - lowest voltage level is length - 2000 ! so simply initalising with last_sample_length-20 doesnt work!
                        encoder_input = torch.cat(
                            [
                                
                                output[
                                    :,
                                    last_sample_length
                                    - self.decoder_input_length
                                    - self.encoder_input_length : last_sample_length
                                    - self.decoder_input_length,
                                ],
                                decoder_input[
                                    :,
                                    last_sample_length
                                    - self.decoder_input_length
                                    - self.encoder_input_length : last_sample_length
                                    - self.decoder_input_length,
                                ],
                            ],
                            dim=-1,
                        )
    
                        decoder_input = curve_samples[-1, :, 1].reshape(1, -1, 1).to(self.comp_mode)
                        self.eval()
                        with torch.inference_mode():
                            output = self(encoder_input, decoder_input)
                        predicted_voltage[-self.decoder_input_length:] = output.squeeze()
                        
                # Denormalize
                # actual_current = (
                #     actual_current * (self.current_max - self.current_min)
                #     + self.current_min
                # )
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
            # val_loss = self.trainer.callback_metrics["val_loss"]
            # combined_val_loss = val_loss + avg_profile_loss
            # Log combined loss
            self.log("pred_loss", avg_profile_loss, on_epoch=True, prog_bar=True)
            
            if self.relax_loss:
                if not self.area_loss_check: # check if model is sufficiently trained for area loss computation
                    if self.trainer.callback_metrics['val_mae_epoch'] < 0.03:
                        self.area_loss_check = True
                        print('model is sufficiently trained -> activate area_loss (OCV spread) computation and logging!!')
                        
            if self.relax_loss and  self.area_loss_check:
                self.on_validation_epoch_end2()
            else:
                self.log("OCV_area_loss", np.nan, on_epoch=True, prog_bar=True)
    
            
    
    def on_validation_epoch_end2(self, SOH_testing=False, relaxation_length = 250,number_of_subsegments = 5 ,dropped_profiles_list=[], additional_curve_dir = None, measurement_data_analysis=False ):
        """
        relaxation pauses of length relaxation_length are added sequentially between the prediction sequences, 
        however the model doesn't really see this, as it is always initalised with the load steps before the terminal relaxation sequence
        --> simulation closely resembles the actual curve from the dataset and added relaxation segments only for OCV evaluation

        Parameters
        ----------

        relaxation_length : TYPE, optional
            DESCRIPTION. The default is 250.

        Returns
        -------
        None.

        """
        index_list = np.arange(1,number_of_subsegments)
        relaxation_result_dict = dict()

        scaled_relax_current =  (np.array([0]) - self.current_min) / ( self.current_max - self.current_min)
        
        if additional_curve_dir:
            additional_curves = self.load_curves(additional_curve_dir)
            self.curves.update(additional_curves)

        for profile, curve in deepcopy(self.curves).items():
            if profile in dropped_profiles_list:
                continue

            Q_OCV_list = []

            # Calculate the residual length for the final sample
            last_sample_length = (curve[self.encoder_input_length:].shape[0] % (self.decoder_input_length- relaxation_length))
            
            # Apply MinMax Normalization -> [0,1]
            curve["voltage"] = (curve["voltage"] - self.voltage_min) / ( self.voltage_max - self.voltage_min)
            curve["current"] = (curve["current"] - self.current_min) / (self.current_max - self.current_min)
            
            
            curve_sliding_samples = sliding_window_view(
                curve[self.encoder_input_length:], (self.decoder_input_length-relaxation_length, 2)
            ).squeeze()

            curve_samples = np.concatenate((
                        curve_sliding_samples[::self.decoder_input_length-relaxation_length],
                        curve_sliding_samples[-1:]),axis=0)

            predicted_voltage = torch.zeros((curve_samples.shape[0]-1)*(self.decoder_input_length)+last_sample_length+relaxation_length, dtype=torch.float)

            encoder_input = (
                torch.tensor(curve[:self.encoder_input_length].to_numpy(), dtype=torch.float)
                .unsqueeze(0)
                .to(self.comp_mode)
            )
            
            decoder_input = curve_samples[0, :, 1]
            actual_current = (curve_samples[0, :, 1]  * (self.current_max - self.current_min) + self.current_min)

            self.eval()
            with torch.inference_mode():
                for index in index_list:
                    current_index = int(index*len(actual_current) /(len(index_list)))
                    Q_OCV_list.append(
                        np.array(
                            [
                                np.cumsum(actual_current[:current_index])[-1],
                                self(
                                    encoder_input,
                                    torch.tensor(
                                        np.concatenate(
                                            (
                                                decoder_input[:current_index],
                                                np.full((relaxation_length), scaled_relax_current),
                                            ),dtype=np.float32
                                        )
                                    ).reshape(1, -1, 1)
                                .to(self.comp_mode))
                                .squeeze()[-1].cpu(),
                            ],
                        )
                    )
            

                # complete decoder length prediction for storage
                relaxated_decoder_input = torch.tensor(np.concatenate((decoder_input, np.full((relaxation_length),scaled_relax_current) ) )
                                                       .reshape(1, -1, 1),dtype=torch.float).to(self.comp_mode)
                output = self(encoder_input, relaxated_decoder_input)
                
            predicted_voltage[:len(output.squeeze())] = output.squeeze()
            
            last_segment_Q_value = Q_OCV_list[-1][0]
            for i in range(1, curve_samples.shape[0]):
                if i != curve_samples.shape[0] - 1:
                    # setup current encoder_input : consisting of last 20 predictions from prev output 
                    encoder_input = torch.cat(
                        [
                            output[:, -self.encoder_input_length-relaxation_length:-relaxation_length,],
                            relaxated_decoder_input[:, -self.encoder_input_length-relaxation_length:-relaxation_length,],
                        ],
                        dim=-1,
                    )
                    # get next current input 
                    decoder_input = curve_samples[i, :, 1]
                    actual_current = (curve_samples[i, :, 1]  * (self.current_max - self.current_min)+ self.current_min)

                    self.eval()
                    with torch.inference_mode():
                        for index in index_list:
                            current_index = int(index*len(actual_current) /(len(index_list)))

                            Q_OCV_list.append(
                                np.array(
                                    [
                                        last_segment_Q_value + np.cumsum(actual_current[:current_index])[-1],
                                        self(
                                            encoder_input,
                                            torch.tensor(
                                                np.concatenate(
                                                    (
                                                        decoder_input[:current_index],
                                                        np.full((relaxation_length), scaled_relax_current),
                                                    ),dtype=np.float32
                                                )
                                            )
                                            .reshape(1, -1, 1)
                                        .to(self.comp_mode))
                                        .squeeze()[-1].cpu(),
                                    ],
                                )
                            )
                        relaxated_decoder_input = torch.tensor(np.concatenate((decoder_input, np.full((relaxation_length),scaled_relax_current) )).reshape(1, -1, 1),dtype=torch.float).to(self.comp_mode)
                        output = self(encoder_input, relaxated_decoder_input)
                        
                    predicted_voltage[ i * (self.decoder_input_length) :  (i + 1) * (self.decoder_input_length)] = output.squeeze()
                    last_segment_Q_value = Q_OCV_list[-1][0]


                else: # last segment residual
                    if last_sample_length>relaxation_length+self.encoder_input_length: # only if leftover segment is really relevant for OCV curve
                        # encoder_input = torch.cat(
                        #     [
                        #         output[:,last_sample_length  - self.decoder_input_length - self.encoder_input_length - relaxation_length: last_sample_length - self.decoder_input_length - relaxation_length],
                        #         relaxated_decoder_input[:,  last_sample_length  - self.decoder_input_length - self.encoder_input_length - relaxation_length : last_sample_length - self.decoder_input_length - relaxation_length],
                        #     ],dim=-1)
                        encoder_input = torch.cat(
                            [
                                output[:,last_sample_length  - self.decoder_input_length - self.encoder_input_length : last_sample_length - self.decoder_input_length ], # removed relaxation_length as decoder_input_length already contains it!
                                relaxated_decoder_input[:,  last_sample_length  - self.decoder_input_length - self.encoder_input_length  : last_sample_length - self.decoder_input_length ],
                            ],dim=-1)

                        decoder_input = curve_samples[-1, :, 1]
                        # actual_current = (curve_samples[i, :, 1][-last_sample_length:] * (self.current_max - self.current_min)+ self.current_min)
                        actual_current = (curve_samples[-1, :, 1]  * (self.current_max - self.current_min)+ self.current_min)

                        index_tracking_list = list() # for Q fixing
                        temp_Q_OCV_list = list()
                        self.eval()
                        with torch.inference_mode():
                            # mod_index_list = np.arange(1, int(number_of_subsegments* last_sample_length/(self.decoder_input_length-relaxation_length))) # elsewise too fine sampling of the last segment
                            for index in index_list:
                                current_index = int(index*len(actual_current) /(len(index_list)))
                                index_tracking_list.append(current_index)
                                temp_Q_OCV_list.append(
                                    np.array(
                                        [
                                            last_segment_Q_value + np.cumsum(actual_current[:current_index])[-1],
                                            self(
                                                encoder_input,
                                                torch.tensor(
                                                    np.concatenate(
                                                        (
                                                            decoder_input[:current_index],
                                                            np.full((relaxation_length), scaled_relax_current),
                                                        ),dtype=np.float32
                                                    )
                                                )
                                                .reshape(1, -1, 1)
                                            .to(self.comp_mode))
                                            .squeeze()[-1].cpu(),
                                        ],
                                    )
                                )
                            
                            relaxated_decoder_input = torch.tensor(np.concatenate((decoder_input, np.full((relaxation_length),scaled_relax_current) )).reshape(1, -1, 1),dtype=torch.float).to(self.comp_mode)
                            output = self(encoder_input, relaxated_decoder_input)

                        valid_last_segment_ids = np.where(np.asarray(index_tracking_list) > len(actual_current)-last_sample_length)[0]
                        temp_Q_OCV_list = np.asarray(temp_Q_OCV_list)[valid_last_segment_ids]
                        # correct Q value for every valid entry in temp_Q_OCV_list, as it is too big due to necessary encoder initalisation with higher voltage levels
                        temp_Q_OCV_list[:,0] = temp_Q_OCV_list[:,0] - np.cumsum(actual_current[:-last_sample_length])[-1]
                        predicted_voltage[-(last_sample_length+relaxation_length):] = output.squeeze()[-(last_sample_length+relaxation_length):]
                        # adjust Q for last_sample_length
                        Q_OCV_list = np.vstack(( np.asarray(Q_OCV_list), temp_Q_OCV_list))
                        
                        
            # for pause analysis with measured curve data
            if measurement_data_analysis:
                real_current = curve['current']* (self.current_max - self.current_min) + self.current_min
                if min(abs(real_current))<0.05:
                    zero_ids = np.where(abs(real_current)<=0.05)[0]
                else:
                    zero_ids = np.where(abs(real_current)<=0.4)[0]

                breaks = np.diff(zero_ids) > 1
                last_indices = zero_ids[np.where(breaks)[0]]  # Indices just before the breaks
                last_indices = np.append(last_indices, zero_ids[-1])
                Q_OCV_list = np.hstack(( np.expand_dims(np.cumsum(real_current)[last_indices],-1), np.expand_dims(curve['voltage'][last_indices],-1) ))
            
            relaxation_result_dict[profile] =  np.asarray(Q_OCV_list)


        
        if not SOH_testing:
            
            points = np.vstack(list(relaxation_result_dict.values()))
            # unscale OCV values
            points[:,1] = points[:,1] * (self.voltage_max - self.voltage_min) + self.voltage_min
            # normalize
            x = points[:, 0]
            y = points[:, 1]
            # scale Q
            x_scaled = (x - x.min()) / (x.max() - x.min())
            
            # Combine scaled x and y back into a single array
            points = np.column_stack((x_scaled, y))
                                            
            alpha = 3 
            concave_hull = alphashape.alphashape(points, alpha)  
    
            if not  isinstance(concave_hull, Polygon) or np.max( concave_hull.exterior.xy[1])< np.max(points[:,1]): # sometimes upper "linear" start segment gets lost, decrease alpha to avoid
                alpha=2
                concave_hull = alphashape.alphashape(points, alpha)  
                if not  isinstance(concave_hull, Polygon) or np.max( concave_hull.exterior.xy[1])< np.max(points[:,1]): # sometimes upper "linear" start segment gets lost, decrease alpha to avoid
                    alpha=1
                    concave_hull = alphashape.alphashape(points, alpha)  
                    if not  isinstance(concave_hull, Polygon) or np.max( concave_hull.exterior.xy[1])< np.max(points[:,1]): # sometimes upper "linear" start segment gets lost, decrease alpha to avoid
                        alpha=0.1
                        concave_hull = alphashape.alphashape(points, alpha)  
                        
            OCV_area_loss = concave_hull.area
            
            # # plot hull
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.scatter(points[:,0],points[:,1])
            # if isinstance(concave_hull, Polygon) :
            #     # Single Polygon
            #     x, y = concave_hull.exterior.xy
            #     plt.plot(x, y, label="Concave Hull", color="red")
            # elif isinstance(concave_hull, MultiPolygon) or np.max( concave_hull.exterior.xy[1])< np.max(points[:,1]): # sometimes upper "linear" start segment gets lost, decrease alpha to avoid
            #     # MultiPolygon
            #     for polygon in concave_hull.geoms:  # Access individual polygons
            #         x, y = polygon.exterior.xy
            #         plt.plot(x, y, label="Concave Hull", color="red")

            # Log combined loss
            self.log("OCV_area_loss", OCV_area_loss, on_epoch=True, prog_bar=True)
            
        else:
            return relaxation_result_dict #, boundary_dict
                
    
    
    
    
    def test_model(self):
        """
        Evaluate the model on test curves using autoregressive prediction.
        This is similar to the on_validation_epoch_end method but adapted for testing.
        Args:
            curves: A dictionary containing test curves (key: profile name, value: curve data).
            self.encoder_input_length: Number of initial data points used for the encoder input.
            self.decoder_input_length: Length of each predicted sequence segment.
        """

    
        # TODO: could be parallelised, so that all 6 profiles are predicted simulaneously?!

        profile_losses = []
        current_profile_dict,voltage_profile_dict,prediction_profile_dict = dict(),dict(),dict()


        for profile, curve in self.curves.items():

            # Calculate the residual length for the final sample
            last_sample_length = (
                curve[self.encoder_input_length:].shape[0] % self.decoder_input_length
            )
            # Apply MinMax Normalization -> [0,1]
            curve["voltage"] = (curve["voltage"] - self.voltage_min) / ( self.voltage_max - self.voltage_min)
            curve["current"] = (curve["current"] - self.current_min) / (self.current_max - self.current_min)
            actual_current = torch.tensor(curve["current"], dtype=torch.float)
            actual_voltage = torch.tensor(curve["voltage"], dtype=torch.float)
            predicted_voltage = torch.zeros(curve.shape[0], dtype=torch.float)
            curve_sliding_samples = sliding_window_view(
                curve[self.encoder_input_length:], (self.decoder_input_length, 2)
            ).squeeze()
            curve_samples = torch.tensor(
                np.concatenate(
                    (
                        curve_sliding_samples[::self.decoder_input_length],
                        curve_sliding_samples[-1:],
                    ),
                    axis=0,
                ),
                dtype=torch.float,
            )
            encoder_input = (
                torch.tensor(curve[:self.encoder_input_length].to_numpy(), dtype=torch.float)
                .unsqueeze(0)
                .to(self.comp_mode)
            )
            # use first 1980 current values as decoder input
            decoder_input = curve_samples[0, :, 1].reshape(1, -1, 1).to(self.comp_mode)
            # start prediction array with inital 20 voltage values 
            predicted_voltage[:self.encoder_input_length] = encoder_input[:, :, 0]
            # necessary to call eval() for prediction mode of model
            self.eval()
            with torch.inference_mode():
                output = self(encoder_input, decoder_input)
            # store output sequence of length 1980 in prediction voltage array (len of initial discharge curve)
            predicted_voltage[
                self.encoder_input_length : self.encoder_input_length + self.decoder_input_length
            ] = output.squeeze()
            for i in range(1, curve_samples.shape[0]):
                # last prediction segment for profile
                if i != curve_samples.shape[0] - 1:
                    # setup current encoder_input : consisting of last 20 predictions from prev output 
                    encoder_input = torch.cat(
                        [
                            output[:, -self.encoder_input_length:],
                            decoder_input[:, -self.encoder_input_length:],
                        ],
                        dim=-1,
                    )
                    # get next current input 
                    decoder_input = curve_samples[i, :, 1].reshape(1, -1, 1).to(self.comp_mode)
                    self.eval()
                    with torch.inference_mode():
                        output = self(encoder_input, decoder_input)
                    # store prediction
                    predicted_voltage[
                        self.encoder_input_length
                        + i * self.decoder_input_length : self.encoder_input_length
                        + (i + 1) * self.decoder_input_length
                    ] = output.squeeze()
                else:
                    # setup this way, as encoder doesn't learn to initalise low voltage levels - lowest voltage level is length - 2000 ! so simply initalising with last_sample_length-20 doesnt work!
                    encoder_input = torch.cat(
                        [
                            
                            output[
                                :,
                                last_sample_length
                                - self.decoder_input_length
                                - self.encoder_input_length : last_sample_length
                                - self.decoder_input_length,
                            ],
                            decoder_input[
                                :,
                                last_sample_length
                                - self.decoder_input_length
                                - self.encoder_input_length : last_sample_length
                                - self.decoder_input_length,
                            ],
                        ],
                        dim=-1,
                    )

                    decoder_input = curve_samples[-1, :, 1].reshape(1, -1, 1).to(self.comp_mode)
                    self.eval()
                    with torch.inference_mode():
                        output = self(encoder_input, decoder_input)
                    predicted_voltage[-self.decoder_input_length:] = output.squeeze()
                    
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

        return current_profile_dict, voltage_profile_dict, prediction_profile_dict, profile_losses


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    
    def load_curves(self,local_curve_dir):
       """Load all curve CSV files into memory."""
           
       curves = {}
       for key, path in local_curve_dir.items():
           curves[key] = pd.read_csv(path / os.listdir(path)[0], usecols=["voltage", "current"])
       return curves
