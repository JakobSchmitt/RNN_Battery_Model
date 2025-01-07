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
from sklearn.pipeline import Pipeline

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

poly_deg = 5 # degree for SOH regr
rem_threshold = 0.02 #  (für aged modell im code verdoppelt!) points to keep : absolute deviation threshold in V for linear model deviation computation -> removal of outlier (dependson num_of_seg)

spread_control = 600 #

measurement_data_analysis = True # for evaluation of pause segments in SOH experiments

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

    additional_data_dirs = [] #['1','3','5','7','9','11','13','15','17' ]
    dropped_profiles_list = []


    if len(additional_data_dirs)>0:
        additional_data_root_dir = Path('C:/Users/s8940173/Dokumente/GitHub/RNN_Battery_Model/Data/2.3_simple_upsampled_20')
        additional_curve_dir = dict(zip(additional_data_dirs,[additional_data_root_dir / dir_id for dir_id in additional_data_dirs]))     # None else  
    else:
        additional_curve_dir = None
    model_1_relaxation_result_dict = model_1.on_validation_epoch_end2(SOH_testing=True,relaxation_length=relax_length,
                                                                                     number_of_subsegments = num_of_seg,
                                                                                     dropped_profiles_list=dropped_profiles_list,
                                                                                     additional_curve_dir = additional_curve_dir,
                                                                                     measurement_data_analysis = measurement_data_analysis) 
    additional_data_dirs =   []
    dropped_profiles_list = [] #['3','5','7','0','2','14','6','8','10','1' ] #,'16','18' 

    if len(additional_data_dirs)>0:
        additional_data_root_dir = Path('C:/Users/s8940173/Dokumente/GitHub/RNN_Battery_Model/Data/2.0_simple_upsampled_20')
        additional_curve_dir = dict(zip(additional_data_dirs,[additional_data_root_dir / dir_id for dir_id in additional_data_dirs]))     # None else  
    else:
        additional_curve_dir = None
    model_2_relaxation_result_dict  = model_2.on_validation_epoch_end2(SOH_testing=True,relaxation_length=relax_length,
                                                                       number_of_subsegments = num_of_seg,
                                                                       dropped_profiles_list=dropped_profiles_list,
                                                                       additional_curve_dir = additional_curve_dir,
                                                                       measurement_data_analysis = measurement_data_analysis) 
    
    test_save_path =  Path(cfg.model_2_pretrained_weights_dir) / "SOH_plots"
    test_save_path.mkdir(parents=True, exist_ok=True)
    
    compute_SOH(model_1_relaxation_result_dict, model_2_relaxation_result_dict, test_save_path,cfg)
    



def compute_SOH(model_1_relaxation_result_dict, model_2_relaxation_result_dict ,save_path,cfg):
    """


    """
    plotting = False
    # points are scaled (Q and V)
    for key in model_1_relaxation_result_dict.keys():
        lin_data = model_1_relaxation_result_dict[key]
        # check if last values are artefacts and have same Q value:
        if plotting:
            plt.figure()
            plt.scatter(lin_data[:,0],lin_data[:,1],label='raw data')
        if np.abs(np.diff(lin_data[-2:,0]))< 1:            # remove all artefacts
            lin_data = lin_data[np.where(np.abs(np.diff(lin_data[:,0]))> 1)[0],:]
        
        Q_range = lin_data[:,0]
        outliers = np.zeros_like(lin_data[:,0], dtype=bool)  # Initialize a boolean mask for outliers
        previous_valid = lin_data[:,:][0]  # Start with the first point as valid
        lin_threshold = np.median(np.diff(lin_data[:,1]))

        for i in range(1, len(lin_data[:,1])):
            if   ( ( lin_data[:,1][i] - previous_valid[1] ) > lin_threshold * (lin_data[:,0][i] - previous_valid[0])/spread_control) or ( (lin_data[:,0][i] - previous_valid[0]) > 1700 ) or ( (lin_data[:,0][i] - previous_valid[0]) < 25 ): 
                outliers[i] = True  # Mark as outlier
            else:
                previous_valid = lin_data[:,:][i]  # Update the reference point if the value is valid
        if min(lin_data[:,1])<0.4:
            degree=5
        else:
            degree=4
        lin_model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())]) 
        
        lin_model.fit(lin_data[~outliers][:,0].reshape(-1,1) ,lin_data[~outliers][:,1])
        if plotting:
            plt.scatter(lin_data[~outliers][:,0],lin_data[~outliers][:,1],label='ohne outlier - lin fit')
        OCV_range = lin_model.predict(Q_range.reshape(-1,1))
        if OCV_range[-1]>OCV_range[-2]:
            # fix wrong model degree
            lin_model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree-1)),
                ('linear', LinearRegression())]) 
            lin_model.fit(lin_data[~outliers][:,0].reshape(-1,1) ,lin_data[~outliers][:,1])
            if plotting:
                plt.scatter(lin_data[~outliers][:,0],lin_data[~outliers][:,1],label='ohne outlier - lin fit')
            OCV_range = lin_model.predict(Q_range.reshape(-1,1))
        if plotting:
            plt.plot(Q_range,OCV_range,color='cyan')
        keeper = np.where( (abs(lin_data[:,1] - OCV_range) < rem_threshold) | ((lin_data[:,1] - OCV_range < 0) &  (abs(lin_data[:,1] - OCV_range) < 2*rem_threshold)) )[0]
        model_1_relaxation_result_dict[key] = lin_data[keeper]
        if plotting:
            plt.scatter(model_1_relaxation_result_dict[key][:,0],model_1_relaxation_result_dict[key][:,1], label='filtered data')
            plt.legend()
            plt.title(f'new - {key}')
        
    model_1_points = np.vstack(list(model_1_relaxation_result_dict.values()))

    model_1_x = model_1_points[:, 0]
    model_1_y_scaled = model_1_points[:, 1]
    # scale Q
    model_1_Q_min = model_1_x.min()
    model_1_Q_max = model_1_x.max()
    model_1_x_scaled = (model_1_x - model_1_Q_min) / (model_1_Q_max - model_1_Q_min)
    
    
    plotting = False
    for key in model_2_relaxation_result_dict.keys():
        lin_data = model_2_relaxation_result_dict[key]
        # check if last values are artefacts and have same Q value:
        if plotting:
            plt.figure()
            plt.scatter(lin_data[:,0],lin_data[:,1],label='raw data')
        if np.abs(np.diff(lin_data[-2:,0]))< 1:            # remove all artefacts
            lin_data = lin_data[np.where(np.abs(np.diff(lin_data[:,0]))> 1)[0],:]
        
        Q_range = lin_data[:,0]
        outliers = np.zeros_like(lin_data[:,0], dtype=bool)  # Initialize a boolean mask for outliers
        previous_valid = lin_data[:,:][0]  # Start with the first point as valid
        lin_threshold = np.median(np.diff(lin_data[:,1]))

        for i in range(1, len(lin_data[:,1])):
            if   ( ( lin_data[:,1][i] - previous_valid[1] ) > lin_threshold * (lin_data[:,0][i] - previous_valid[0])/spread_control ) or ( (lin_data[:,0][i] - previous_valid[0]) > 1700 ) or ( (lin_data[:,0][i] - previous_valid[0]) < 25 ): 
                outliers[i] = True  # Mark as outlier
            else:
                previous_valid = lin_data[:,:][i]  # Update the reference point if the value is valid
            
        if min(lin_data[:,1])<0.4:
            degree=5
        else:
            degree=3
        lin_model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())]) 
        
        lin_model.fit(lin_data[~outliers][:,0].reshape(-1,1) ,lin_data[~outliers][:,1]) #int(len(lin_data[~outliers])/4)
        if plotting:
            plt.scatter(lin_data[~outliers][:,0],lin_data[~outliers][:,1],label='ohne outlier - lin fit')
        OCV_range = lin_model.predict(Q_range.reshape(-1,1))
        if OCV_range[-1]>OCV_range[-2]:
            # fix wrong model degree
            lin_model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree-1)),
                ('linear', LinearRegression())]) 
            lin_model.fit(lin_data[~outliers][:,0].reshape(-1,1) ,lin_data[~outliers][:,1])
            if plotting:
                plt.scatter(lin_data[~outliers][:,0],lin_data[~outliers][:,1],label='ohne outlier - lin fit')
            OCV_range = lin_model.predict(Q_range.reshape(-1,1))
        if plotting:
            plt.plot(Q_range,OCV_range,color='cyan')
            

        keeper = np.where( (abs(lin_data[:,1] - OCV_range) < rem_threshold*2) | ((lin_data[:,1] - OCV_range < 0) &  (abs(lin_data[:,1] - OCV_range) < 2*rem_threshold*2)) )[0]
        model_2_relaxation_result_dict[key] = lin_data[keeper]
        if plotting:
            plt.scatter(model_2_relaxation_result_dict[key][:,0],model_2_relaxation_result_dict[key][:,1], label='filtered data')
            plt.legend()
            plt.title(f'aged - {key}')    
        
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


    model_1_regr = Pipeline([
        ('poly', PolynomialFeatures(degree=poly_deg)),
        ('linear', LinearRegression())]) 
    model_1_regr.fit(model_1_y_scaled.reshape(-1, 1), model_1_x_scaled )
    model_1_regr_predictions_scaled = model_1_regr.predict(y_scaled_range.reshape(-1, 1))
    
    
    model_1_regr_predictions =  model_1_regr_predictions_scaled * (model_1_Q_max - model_1_Q_min) + model_1_Q_min
    model_1_y_unscaled = model_1_y_scaled  * (cfg.method.voltage_max - cfg.method.voltage_min) + cfg.method.voltage_min 
    model_1_x_unscaled = model_1_x_scaled  * (model_1_Q_max - model_1_Q_min) + model_1_Q_min
    


    model_2_regr = Pipeline([
        ('poly', PolynomialFeatures(degree=poly_deg)),
        ('linear', LinearRegression())]) 
    model_2_regr.fit(model_2_y_scaled.reshape(-1, 1), model_2_x_scaled )
    model_2_regr_predictions_scaled = model_2_regr.predict( y_scaled_range.reshape(-1, 1))
    
    model_2_regr_predictions =  model_2_regr_predictions_scaled  * (model_2_Q_max - model_2_Q_min) + model_2_Q_min
    model_2_y_unscaled = model_2_y_scaled* (cfg.method.voltage_max - cfg.method.voltage_min) + cfg.method.voltage_min 
    model_2_x_unscaled = model_2_x_scaled * (model_2_Q_max - model_2_Q_min) + model_2_Q_min
    
    
    # num_neighbors = num_of_seg-13
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
