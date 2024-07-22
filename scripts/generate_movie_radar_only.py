import sys
import os
import numpy as np
from CPSL_Radar.Analyzer import Analyzer
from CPSL_Radar.datasets.Dataset_Generator import DatasetGenerator
from CPSL_Radar.models.unet import unet
from torchvision import transforms

def main():
    
    dataset_generator = init_dataset_generator(generate_dataset=False)

    #initialize the transforms
    input_transforms = [
        transforms.ToTensor(),
        transforms.Resize((64,48))
    ]

   #initialize the unet
    unet_model = unet(
        encoder_input_channels= 40,
        encoder_out_channels= (64,128,256),
        decoder_input_channels= (512,256,128),
        decoder_out_channels= 64,
        output_channels= 1,
        retain_dimmension= False,
        input_dimmensions= (64,48)
    )

    #initialize the viewer
    viewer = Analyzer(
        dataset_generator=dataset_generator,
        model=unet_model,
        transforms_to_apply= input_transforms,
        working_dir="../working_dir/",
        model_state_dict_file_name="RadCloud_40_chirps_10e.pth",
        cuda_device="cuda:0"
    )

    viewer.save_video("RadCloud_40_chirps_10e.mp4",fps=10)



def init_dataset_generator(generate_dataset = False):

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
    scenarios_to_use = ugv_seen_test_scenarios

    #location that we wish to save the dataset to
    generated_dataset_path = "../data/test/"

    #specifying the names for the files
    generated_file_name = "frame"
    radar_data_folder = "radar"
    lidar_data_folder = None

    #basic dataset settings
    num_chirps_to_save = 40
    num_previous_frames = 0

    #initialize the DatasetGenerator
    dataset_generator = DatasetGenerator(radar_data_only=True)

    dataset_generator.config_generated_dataset_paths(
        generated_dataset_path=generated_dataset_path,
        generated_file_name=generated_file_name,
        generated_radar_data_folder=radar_data_folder,
        generated_lidar_data_folder=None,
        clear_existing_data=generate_dataset
    )

    #configure the lidar data processor
    dataset_generator.config_radar_lidar_data_paths(
        scenario_folder= scenarios_to_use[0],
        radar_data_folder=radar_data_folder,
        lidar_data_folder=None
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

    if generate_dataset:
        dataset_generator.generate_dataset_from_multiple_scenarios(
            scenario_folders = scenarios_to_use,
            radar_data_folder= radar_data_folder,
            lidar_data_folder=None
        )
    
    return dataset_generator

if __name__ == '__main__':
    
    #change directory if needed
    # os.chdir("..")

    main()
    sys.exit()