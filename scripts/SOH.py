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
from sklearn.neighbors import KNeighborsRegressor

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

num_of_seg = 15 # num of segments for relax point creation within a decoder prediction length
relax_length = 200 # length of relax segment

poly_deg = 3# degree for SOH regr
rem_threshold = 0.02 # absolute deviation threshold in V for linear model deviation computation -> removal of outlier (dependson num_of_seg)
lin_model_fac = 0.5 # factor for lin model creation

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

    dropped_profiles_list = []
    model_1_relaxation_result_dict = model_1.on_validation_epoch_end2(SOH_testing=True,relaxation_length=relax_length,
                                                                                     number_of_subsegments = num_of_seg,
                                                                                     dropped_profiles_list=dropped_profiles_list) 

    dropped_profiles_list = ['3','5','7','0','2','14','6','8','10','1' ] #,'16','18' 

    model_2_relaxation_result_dict  = model_2.on_validation_epoch_end2(SOH_testing=True,relaxation_length=relax_length,
                                                                       number_of_subsegments = num_of_seg,
                                                                       dropped_profiles_list=dropped_profiles_list)

    
    test_save_path =  Path(cfg.model_2_pretrained_weights_dir) / "SOH_plots"
    test_save_path.mkdir(parents=True, exist_ok=True)
    
    compute_SOH(model_1_relaxation_result_dict, model_2_relaxation_result_dict, test_save_path,cfg)
    



def compute_SOH(model_1_relaxation_result_dict, model_2_relaxation_result_dict ,save_path,cfg):
    """


    """
    
    # points are scaled (Q and V)
    for key in model_1_relaxation_result_dict.keys():
        lin_data = model_1_relaxation_result_dict[key]
        # plt.figure()
        # plt.scatter(lin_data[:,0],lin_data[:,1])
        Q_range = lin_data[:,0]
        outliers = np.zeros_like(lin_data[:,0], dtype=bool)  # Initialize a boolean mask for outliers
        previous_valid = lin_data[:,:][0]  # Start with the first point as valid
        lin_threshold = np.median(np.diff(lin_data[:,1]))*lin_model_fac

        for i in range(1, len(lin_data[:,1])):
            if  lin_data[:,1][i] - previous_valid[1] > lin_threshold * (lin_data[:,0][i] - previous_valid[0])/100: #handling of very large Q gaps and slightly lower OCV values, prev not recognised
                outliers[i] = True  # Mark as outlier
            else:
                previous_valid = lin_data[:,:][i]  # Update the reference point if the value is valid
        
        lin_model = LinearRegression()
        lin_model.fit(lin_data[~outliers][:,0].reshape(-1,1) ,lin_data[~outliers][:,1])
        # plt.scatter(lin_data[~outliers][:,0],lin_data[~outliers][:,1])
        OCV_range = lin_model.predict(Q_range.reshape(-1,1))
        keeper = np.where( (abs(lin_data[:,1] - OCV_range) < rem_threshold) | ((lin_data[:,1] - OCV_range < 0) &  (abs(lin_data[:,1] - OCV_range) < 2*rem_threshold)) )[0]
        
        model_1_relaxation_result_dict[key] = lin_data[keeper]
        # plt.scatter(model_1_relaxation_result_dict[key][:,0],model_1_relaxation_result_dict[key][:,1])
        # print('')

        
    model_1_points = np.vstack(list(model_1_relaxation_result_dict.values()))

    model_1_x = model_1_points[:, 0]
    model_1_y_scaled = model_1_points[:, 1]
    # scale Q
    model_1_Q_min = model_1_x.min()
    model_1_Q_max = model_1_x.max()
    model_1_x_scaled = (model_1_x - model_1_Q_min) / (model_1_Q_max - model_1_Q_min)
    
    
    
    for key in model_2_relaxation_result_dict.keys():
        lin_data = model_2_relaxation_result_dict[key]
        plt.figure()
        plt.scatter(lin_data[:,0],lin_data[:,1])
        Q_range = lin_data[:,0]
        outliers = np.zeros_like(lin_data[:,0], dtype=bool)  # Initialize a boolean mask for outliers
        previous_valid = lin_data[:,:][0]  # Start with the first point as valid
        lin_threshold = np.median(np.diff(lin_data[:,1]))*lin_model_fac

        for i in range(1, len(lin_data[:,1])):
            if  lin_data[:,1][i] - previous_valid[1] > lin_threshold * (lin_data[:,0][i] - previous_valid[0])/100:
                outliers[i] = True  # Mark as outlier
            else:
                previous_valid = lin_data[:,:][i]  # Update the reference point if the value is valid
        
        lin_model = LinearRegression()
        lin_model.fit(lin_data[~outliers][:,0].reshape(-1,1) ,lin_data[~outliers][:,1]) #int(len(lin_data[~outliers])/4)
        # plt.scatter(lin_data[~outliers][:,0],lin_data[~outliers][:,1])
        OCV_range = lin_model.predict(Q_range.reshape(-1,1))
        keeper = np.where( (abs(lin_data[:,1] - OCV_range) < rem_threshold) | ((lin_data[:,1] - OCV_range < 0) &  (abs(lin_data[:,1] - OCV_range) < 2*rem_threshold)) )[0]

        # keeper = np.where(abs(lin_data[:,1] - OCV_range) < rem_threshold)[0]
        model_2_relaxation_result_dict[key] = lin_data[keeper]
        # plt.scatter(model_2_relaxation_result_dict[key][:,0],model_2_relaxation_result_dict[key][:,1])
    
    model_2_points = np.vstack(list(model_2_relaxation_result_dict.values()))
    model_2_x = model_2_points[:, 0]
    model_2_y_scaled = model_2_points[:, 1]
    # scale Q
    model_2_Q_min = model_2_x.min()
    model_2_Q_max = model_2_x.max()
    model_2_x_scaled = (model_2_x - model_2_Q_min) / (model_2_Q_max - model_2_Q_min)
    
    scaled_comb_y_max =  min(max(model_1_y_scaled),max(model_2_y_scaled)) 
    # scaled_comb_y_max = (3.8 - cfg.method.voltage_min) / (cfg.method.voltage_max - cfg.method.voltage_min) #
    scaled_comb_y_min = max(min(model_1_y_scaled),min(model_2_y_scaled)) # 
    # scaled_comb_y_min = (3.6 - cfg.method.voltage_min) / (cfg.method.voltage_max - cfg.method.voltage_min) # 
    y_scaled_range = np.linspace(scaled_comb_y_min,scaled_comb_y_max,2000)
    y_range = y_scaled_range*(cfg.method.voltage_max - cfg.method.voltage_min) +  cfg.method.voltage_min

    
    # model 1 - new
    model_1_poly = PolynomialFeatures(degree=poly_deg).fit(model_1_y_scaled.reshape(-1, 1))
    model_1_y_scaled_range_poly = model_1_poly.transform(y_scaled_range.reshape(-1, 1))
    # regression
    model_1_regr = LinearRegression()
    model_1_regr.fit(model_1_poly.transform(model_1_y_scaled.reshape(-1, 1)),  model_1_x_scaled)
    model_1_regr_predictions_scaled = model_1_regr.predict(model_1_y_scaled_range_poly)
    model_1_regr_predictions =  model_1_regr_predictions_scaled * (model_1_Q_max - model_1_Q_min) + model_1_Q_min
    model_1_y_unscaled = model_1_y_scaled  * (cfg.method.voltage_max - cfg.method.voltage_min) + cfg.method.voltage_min 
    model_1_x_unscaled = model_1_x_scaled  * (model_1_Q_max - model_1_Q_min) + model_1_Q_min
    
    # model 2 - aged
    model_2_poly = PolynomialFeatures(degree=poly_deg).fit(model_2_y_scaled.reshape(-1, 1))
    model_2_y_scaled_range_poly = model_2_poly.transform(y_scaled_range.reshape(-1, 1))
    # regression
    model_2_regr = LinearRegression()
    model_2_regr.fit(model_2_poly.transform(model_2_y_scaled.reshape(-1, 1)), model_2_x_scaled )
    model_2_regr_predictions_scaled = model_2_regr.predict( model_2_y_scaled_range_poly)
    model_2_regr_predictions =  model_2_regr_predictions_scaled  * (model_2_Q_max - model_2_Q_min) + model_2_Q_min
    model_2_y_unscaled = model_2_y_scaled* (cfg.method.voltage_max - cfg.method.voltage_min) + cfg.method.voltage_min 
    model_2_x_unscaled = model_2_x_scaled * (model_2_Q_max - model_2_Q_min) + model_2_Q_min
    
    
    # num_neighbors = num_of_seg
    # knn_1 = KNeighborsRegressor(n_neighbors=num_neighbors)
    # knn_1.fit(model_1_y_scaled.reshape(-1, 1) , model_1_x_scaled )
    # model_1_regr_predictions  = knn_1.predict(y_scaled_range.reshape(-1, 1)) * (model_1_Q_max - model_1_Q_min) + model_1_Q_min

    # knn_2 = KNeighborsRegressor(n_neighbors=num_neighbors)
    # knn_2.fit(model_2_y_scaled.reshape(-1, 1) , model_2_x_scaled )
    # model_2_regr_predictions  = knn_2.predict(y_scaled_range.reshape(-1, 1)) * (model_2_Q_max - model_2_Q_min) + model_2_Q_min


    SOH = model_2_regr_predictions / model_1_regr_predictions
    SOH_mean = np.full((SOH.shape),np.mean(SOH))
    lower_SOH_id = np.where(y_range < 3.9)[0][-1]
    SOH_mean_lower = np.full((lower_SOH_id),np.mean(SOH[:lower_SOH_id]))
    #
    plt.figure(figsize=(10, 6))
    plt.scatter(model_1_x_unscaled, model_1_y_unscaled,s=20,color='blue')
    plt.scatter(model_2_x_unscaled, model_2_y_unscaled,s=20,color='green')
    plt.plot(model_1_regr_predictions,y_range,c='blue')
    plt.plot(model_2_regr_predictions,y_range,c="green")
    plt.tight_layout()
    # plt.savefig(save_path/".png")  # Save each profile as a figure
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(y_range,SOH,linewidth=3)
    plt.plot(y_range,SOH_mean,linewidth=2)
    plt.plot(y_range[:lower_SOH_id],SOH_mean_lower,linewidth=2)
    plt.ylim([0.7, 1.05])



if __name__ == "__main__":
    main()
