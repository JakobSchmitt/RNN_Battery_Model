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

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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

dropped_profiles_list = list() # controlls which profile ids to be neglected in SOH computation
@hydra.main(version_base="1.1", config_path="../config/", config_name="SOH")
def main(cfg):

    ### model 1 loading ###
    model_1_pretrained_weights_path = Path(cfg.model_1_pretrained_weights_dir) / 'weights' / 'all_epochs' 
    model_1_all_epoch_files = list(model_1_pretrained_weights_path.glob('*.ckpt'))
    model_1_all_epoch_files = sorted(model_1_all_epoch_files, key=os.path.getctime)
    model_1_path = str(model_1_all_epoch_files[cfg.model_1_id])
    model_1 = EncoderDecoderGRU.load_from_checkpoint(Path(model_1_path))
    logging.info(f"{cfg.method.mode} : Model 1 loaded using checkpoint")
    
    ### model 2 loading ###
    model_2_pretrained_weights_path = Path(cfg.model_2_pretrained_weights_dir) / 'weights' / 'all_epochs' 
    model_2_all_epoch_files = list(model_2_pretrained_weights_path.glob('*.ckpt'))
    model_2_all_epoch_files = sorted(model_2_all_epoch_files, key=os.path.getctime)
    model_2_path = str(model_2_all_epoch_files[cfg.model_2_id])
    model_2 = EncoderDecoderGRU.load_from_checkpoint(Path(model_2_path))
    logging.info(f"{cfg.method.mode} : Model 2 loaded using checkpoint")

    # dropped_profiles_list = ['2','1','3','5']
    dropped_profiles_list = ['2']

    model_1_relaxation_result_dict = model_1.on_validation_epoch_end2(SOH_testing=True,relaxation_length=100,number_of_subsegments = 15,dropped_profiles_list=dropped_profiles_list)
    
    # dropped_profiles_list = ['0','5','8','1','4','3'] # 129_stoch
    # dropped_profiles_list = ['0','7','12','1','6']
    dropped_profiles_list = ['5']

    model_2_relaxation_result_dict  = model_2.on_validation_epoch_end2(SOH_testing=True,relaxation_length=100,number_of_subsegments = 15,dropped_profiles_list=dropped_profiles_list)

    
    test_save_path =  Path(cfg.model_2_pretrained_weights_dir) / "SOH_plots"
    test_save_path.mkdir(parents=True, exist_ok=True)
    
    compute_SOH(model_1_relaxation_result_dict, model_2_relaxation_result_dict, test_save_path,cfg)
    



def compute_SOH(model_1_relaxation_result_dict, model_2_relaxation_result_dict ,save_path,cfg):
    """


    """
    poly_deg = 9
    # points are scaled (Q and V)
    model_1_points = np.vstack(list(model_1_relaxation_result_dict.values()))
    model_1_x = model_1_points[:, 0]
    model_1_y_scaled = model_1_points[:, 1]
    # scale Q
    model_1_Q_min = model_1_x.min()
    model_1_Q_max = model_1_x.max()
    model_1_x_scaled = (model_1_x - model_1_Q_min) / (model_1_Q_max - model_1_Q_min)
    
    model_2_points = np.vstack(list(model_2_relaxation_result_dict.values()))
    model_2_x = model_2_points[:, 0]
    model_2_y_scaled = model_2_points[:, 1]
    # scale Q
    model_2_Q_min = model_2_x.min()
    model_2_Q_max = model_2_x.max()
    model_2_x_scaled = (model_2_x - model_2_Q_min) / (model_2_Q_max - model_2_Q_min)
    
    scaled_comb_y_max = (4.0 - cfg.method.voltage_min) / (cfg.method.voltage_max - cfg.method.voltage_min) # min(max(model_1_y_scaled),max(model_2_y_scaled))
    scaled_comb_y_min = (3.35 - cfg.method.voltage_min) / (cfg.method.voltage_max - cfg.method.voltage_min) # max(min(model_1_y_scaled),min(model_2_y_scaled))
    y_scaled_range = np.linspace(scaled_comb_y_min,scaled_comb_y_max,2000)
    y_range = y_scaled_range*(cfg.method.voltage_max - cfg.method.voltage_min) +  cfg.method.voltage_min

    
    # model 1
    model_1_poly = PolynomialFeatures(degree=poly_deg).fit(model_1_y_scaled.reshape(-1, 1))
    model_1_y_scaled_range_poly = model_1_poly.transform(y_scaled_range.reshape(-1, 1))
    # regression
    model_1_regr = LinearRegression()
    model_1_regr.fit(model_1_poly.transform(model_1_y_scaled.reshape(-1, 1)),  model_1_x_scaled)
    model_1_regr_predictions_scaled = model_1_regr.predict(model_1_y_scaled_range_poly)
    model_1_regr_predictions =  model_1_regr_predictions_scaled * (model_1_Q_max - model_1_Q_min) + model_1_Q_min
    model_1_y_unscaled = model_1_y_scaled  * (cfg.method.voltage_max - cfg.method.voltage_min) + cfg.method.voltage_min 
    model_1_x_unscaled = model_1_x_scaled  * (model_1_Q_max - model_1_Q_min) + model_1_Q_min
    
    # model 2
    model_2_poly = PolynomialFeatures(degree=poly_deg).fit(model_2_y_scaled.reshape(-1, 1))
    model_2_y_scaled_range_poly = model_2_poly.transform(y_scaled_range.reshape(-1, 1))
    # regression
    model_2_regr = LinearRegression()
    model_2_regr.fit(model_2_poly.transform(model_2_y_scaled.reshape(-1, 1)), model_2_x_scaled )
    model_2_regr_predictions_scaled = model_2_regr.predict( model_2_y_scaled_range_poly)
    model_2_regr_predictions =  model_2_regr_predictions_scaled  * (model_2_Q_max - model_2_Q_min) + model_2_Q_min
    model_2_y_unscaled = model_2_y_scaled* (cfg.method.voltage_max - cfg.method.voltage_min) + cfg.method.voltage_min 
    model_2_x_unscaled = model_2_x_scaled * (model_2_Q_max - model_2_Q_min) + model_2_Q_min
    
    SOH = model_2_regr_predictions / model_1_regr_predictions
    SOH_mean = np.full((SOH.shape),np.mean(SOH))
    
    #
    plt.figure(figsize=(10, 6))
    plt.scatter(model_1_x_unscaled, model_1_y_unscaled,s=3,color='blue')
    plt.scatter(model_2_x_unscaled, model_2_y_unscaled,s=3,color='green')
    plt.plot(model_1_regr_predictions,y_range,c='blue')
    plt.plot(model_2_regr_predictions,y_range,c="green")
    plt.tight_layout()
    # plt.savefig(save_path/".png")  # Save each profile as a figure
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(y_range,SOH,linewidth=3)
    plt.plot(y_range,SOH_mean,linewidth=2)



if __name__ == "__main__":
    main()
