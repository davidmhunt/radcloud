import sys
sys.path.append("..")
import os
# os.chdir("..")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import CPSL Radar Dataset Generator Code
from radcloud.datasets.Dataset_Generator import DatasetGenerator

dataset_folder = "../data"

#list the scenarios for all datasets
ugv_seen_train_scenarios = ["scene_{}".format(i+1) for i in range(7)]
ugv_seen_test_scenarios = ["scene_{}_test".format(i+1) for i in range(7)]
ugv_unseen_test_scenarios = ["scene_{}".format(i+1) for i in range(7)]
ugv_rapid_movement_test_scenarios = ["scene_{}_test_spin".format(i) for i in range(3,5)]

#generate the full path to each dataset
ugv_seen_train_scenarios = [os.path.join(
    dataset_folder,"ugv_seen_dataset",scenario_folder) for 
    scenario_folder in ugv_seen_train_scenarios]
ugv_seen_test_scenarios = [os.path.join(
    dataset_folder,"ugv_seen_dataset",scenario_folder) for 
    scenario_folder in ugv_seen_test_scenarios]
ugv_unseen_test_scenarios = [os.path.join(
    dataset_folder,"ugv_unseen_dataset",scenario_folder) for 
    scenario_folder in ugv_unseen_test_scenarios]
ugv_rapid_movement_test_scenarios = [os.path.join(
    dataset_folder,"ugv_rapid_movement_dataset",scenario_folder) for 
    scenario_folder in ugv_rapid_movement_test_scenarios]


#select which scenario to generate the dataset from
scenarios_to_use = ugv_seen_train_scenarios

#location that we wish to save the dataset to
generated_dataset_path = "../data/train/"

#specifying the names for the files
generated_file_name = "frame"
radar_data_folder = "radar"
lidar_data_folder = "lidar"

#basic dataset settings
num_chirps_to_save = 40
num_previous_frames = 0

#initialize the DatasetGenerator
dataset_generator = DatasetGenerator()

dataset_generator.config_generated_dataset_paths(
    generated_dataset_path=generated_dataset_path,
    generated_file_name=generated_file_name,
    generated_radar_data_folder=radar_data_folder,
    generated_lidar_data_folder=lidar_data_folder,
    clear_existing_data=True
)

#configure the lidar data processor
dataset_generator.config_radar_lidar_data_paths(
    scenario_folder= scenarios_to_use[0],
    radar_data_folder=radar_data_folder,
    lidar_data_folder=lidar_data_folder
)

#configure the radar data processor
dataset_generator.config_radar_data_processor(
    max_range_bin=64,
    num_chirps_to_save=num_chirps_to_save,
    num_previous_frames=num_previous_frames,
    radar_fov= [-0.87, 0.87], #+/- 50 degrees
    num_angle_bins=64,
    power_range_dB=[60,105],
    chirps_per_frame= 64,
    rx_channels = 4,
    tx_channels = 1,
    samples_per_chirp = 64,
    adc_sample_rate_Hz = 2e6,
    chirp_slope_MHz_us= 35,
    start_freq_Hz=77e9,
    idle_time_us = 100,
    ramp_end_time_us = 100
)

#configure the lidar data processor
dataset_generator.config_lidar_data_processor(
    max_range_m=8.56,
    num_range_bins=64,
    angle_range_rad=[-np.pi/2 - 0.87,-np.pi/2 + 0.87], #[-np.pi /2 , np.pi /2],
    num_angle_bins=48,
    num_previous_frames=num_previous_frames
)

dataset_generator.generate_dataset_from_multiple_scenarios(
    scenario_folders = scenarios_to_use,
    radar_data_folder= radar_data_folder,
    lidar_data_folder=lidar_data_folder
)