# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:13:12 2024

@author: s8940173
"""
import hydra
import logging
from pathlib import Path
import numpy as np
from RNN.datamodules import UniDataModule
from RNN.models import EncoderDecoderGRU

import lightning as L
from lightning.pytorch.loggers import WandbLogger

import matplotlib.pyplot as plt
import os
import torch
torch.set_float32_matmul_precision('high')

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error
)


@hydra.main(version_base="1.1", config_path="../config/", config_name="test")
def main(cfg):

    
    pretrained_weights_path = Path(cfg.pretrained_weights_dir) / 'weights' / 'all_epochs' 
    all_epoch_files = list(pretrained_weights_path.glob('*.ckpt'))
    all_epoch_files = sorted(all_epoch_files, key=os.path.getctime)
    model_path = str(all_epoch_files[cfg.model_id])
    
    model = EncoderDecoderGRU.load_from_checkpoint(Path(model_path))
    logging.info(f"{cfg.method.mode} : Model loaded using checkpoint")

    

    current_profile_dict, voltage_profile_dict, prediction_profile_dict, profile_mae_loss = model.test_model()
    test_save_path =  Path(cfg.pretrained_weights_dir) / "test_plots"
    test_save_path.mkdir(parents=True, exist_ok=True)
    
    plot_voltage_profiles(current_profile_dict, voltage_profile_dict, prediction_profile_dict, profile_mae_loss,test_save_path)

    # # Optionally save predictions
    # with open("test_results.json", "w") as f:
    #     json.dump(predictions, f)
    
    


def plot_voltage_profiles(current_profile_dict, voltage_profile_dict, prediction_profile_dict, profile_mae_loss,save_path):
    """
    Plots the predicted and actual voltage profiles for each profile in separate plots and combined.
    Args:
        current_profile_dict: Dictionary of current values for each profile.
        voltage_profile_dict: Dictionary of actual voltage values for each profile.
        prediction_profile_dict: Dictionary of predicted voltage values for each profile.
        avg_profile_mae_loss: Average mean absolute error loss across profiles.
    """
    plot_separate = False
    # Individual plots
    MAE_list = list()
    for idx,profile in enumerate(voltage_profile_dict.keys()):
        actual_voltage = voltage_profile_dict[profile]
        predicted_voltage = prediction_profile_dict[profile]


        
        r2 = r2_score(actual_voltage, predicted_voltage)
        rmse = np.sqrt(mean_squared_error(actual_voltage, predicted_voltage))
        mae = mean_absolute_error(actual_voltage, predicted_voltage)
        mape = mean_absolute_percentage_error(actual_voltage, predicted_voltage)
        MAE_list.append(mae*1000)
        
        if plot_separate:
            plt.figure(figsize=(10, 6))
            plt.plot(actual_voltage, label="Actual Voltage", color="blue", linewidth=2)
            plt.plot(predicted_voltage, label="Predicted Voltage", color="red", linestyle="--", linewidth=2)
            plt.title(f"Voltage Profile:  R2 : {r2:.5}  RMSE: {rmse:.5f} \n MAE : {mae:.5f}  MAPE : {mape:.5f}", fontsize=14)
            plt.xlabel("Time Step", fontsize=12)
            plt.ylabel("Voltage (V)", fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            # plt.savefig(save_path/f"{profile}_voltage_profile.png")  # Save each profile as a figure
            plt.show()

    # Combined plot
    plt.figure(figsize=(12, 8))
    for idx, profile in enumerate(voltage_profile_dict.keys()):
        actual_voltage = voltage_profile_dict[profile]
        predicted_voltage = prediction_profile_dict[profile]

        plt.plot(actual_voltage, label=f"Actual {profile}", alpha=0.7, linewidth=1.5)
        plt.plot(predicted_voltage, label=f"Predicted {profile} MAE {MAE_list[idx]:.1f}mV", alpha=0.7, linestyle="--", linewidth=1.5)

    plt.title("Combined Voltage Profiles (Actual vs. Predicted)", fontsize=16)
    plt.xlabel("Time Step", fontsize=14)
    plt.ylabel("Voltage (V)", fontsize=14)
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # plt.savefig(save_path/"combined_voltage_profiles.png")  # Save the combined figure
    plt.show()


if __name__ == "__main__":
    main()
