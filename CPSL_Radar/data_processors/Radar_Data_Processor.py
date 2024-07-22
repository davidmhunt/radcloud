import os
import numpy as np
from scipy.constants import c,pi
from scipy.io import loadmat
import matplotlib.pyplot as plt
import imageio
import io
from IPython.display import display, clear_output
from tqdm import tqdm

class RadarDataProcessor:
    
    #global params

    #plotting parameters
    font_size_title = 18
    font_size_axis_labels = 18
    font_size_ticks = 16
    font_size_color_bar = 10
    line_width_axis = 2.5

    def __init__(self):
        
        #given radar parameters
        self.chirp_loops_per_frame = None
        self.rx_channels = None
        self.tx_channels = None
        self.samples_per_chirp = None
        self.adc_sample_rate_Hz = None
        self.chirp_slope_MHz_us = None
        self.start_freq_Hz = None
        self.idle_time_us = None
        self.ramp_end_time_us = None

        #computed radar parameters
        self.chirp_BW_Hz = None

        #computed radar performance specs
        self.range_res = None
        self.range_bins = None
        self.phase_shifts = None
        self.angle_bins = None
        self.angle_bins_to_keep = None
        self.thetas = None
        self.rhos = None
        self.x_s = None
        self.y_s = None

        #raw radar cube for a single frame (indexed by [rx channel, sample, chirp])
        #TODO: add support for DCA1000 Data Processing

        #relative paths of raw radar ADC data
        self.radar_data_paths:list = None
        self.save_file_folder:str = None
        self.save_file_name:str = None
        self.save_file_start_offset = 0 #offset for the save file index if adding data to existing dataset
        self.save_file_number_offset = 10000 #offset for the save file names so that they can be ordered as a sorted list

        
        #plotting
        self.fig = None
        self.axs = None

        self.max_range_bin = 0
        self.num_chirps_to_save = 0
        self.num_angle_bins = 0
        self.power_range_dB = None #specified as [min,max]


        #for taking into account previous frames in the radar data
        self.num_previous_frames = 0

        #to average all of the data together
        self.use_average_range_az = False
        return

    def configure(self,
                    max_range_bin:int,
                    num_chirps_to_save:int,
                    radar_fov:list,
                    num_angle_bins:int,
                    power_range_dB:list,
                    chirps_per_frame,
                    rx_channels,
                    tx_channels,
                    samples_per_chirp,
                    adc_sample_rate_Hz,
                    chirp_slope_MHz_us,
                    start_freq_Hz,
                    idle_time_us,
                    ramp_end_time_us,
                    num_previous_frames=0,
                    use_average_range_az = False):
        
        #load the radar parameters
        self.max_range_bin = max_range_bin
        self.num_chirps_to_save = num_chirps_to_save
        self.radar_fov = radar_fov
        self.num_angle_bins = num_angle_bins
        self.power_range_dB = power_range_dB
        self.chirp_loops_per_frame = chirps_per_frame
        self.rx_channels = rx_channels
        self.tx_channels = tx_channels
        self.samples_per_chirp = samples_per_chirp
        self.adc_sample_rate_Hz = adc_sample_rate_Hz
        self.chirp_slope_MHz_us = chirp_slope_MHz_us
        self.start_freq_Hz = start_freq_Hz
        self.idle_time_us = idle_time_us
        self.ramp_end_time_us = ramp_end_time_us
        

        #init computed params
        self._init_computed_params()

        #print the max range
        print("max range: {}m".format(self.max_range_bin * self.range_res))
        print("num actual angle bins: {}".format(np.sum(self.angle_bins_to_keep)))

        #take previous frames instead of previous chirps into account
        self.num_previous_frames = num_previous_frames

        #for whether or not to average everything
        self.use_average_range_az = use_average_range_az

        return

    def init_radar_data_paths(self,
                        radar_data_paths:list):
        """Initialize the path to each of the samples used to compute the radar data

        Args:
            radar_data_paths (list): The path to the radar data
        """
        self.radar_data_paths = radar_data_paths
        return
    
    def init_save_file_paths(self,
                             save_file_folder:str,
                             save_file_name: str):
        """Initialize the paths to where the generated dataset samples will be saved

        Args:
            save_file_folder (str): path to the folder where the data is to be saved
            save_file_name (str): name of the file that the generated dataset is saved as
        """
        self.save_file_folder = save_file_folder
        self.save_file_name = save_file_name

        return

    def _init_computed_params(self):

        #chirp BW
        self.chirp_BW_Hz = self.chirp_slope_MHz_us * 1e12 * self.samples_per_chirp / self.adc_sample_rate_Hz

        #range resolution
        self.range_res = c / (2 * self.chirp_BW_Hz)
        self.range_bins = np.arange(0,self.samples_per_chirp) * self.range_res

        #angular parameters
        self.phase_shifts = np.arange(pi,-pi  - 2 * pi /(self.num_angle_bins - 1),-2 * pi / (self.num_angle_bins-1))
        #round the last entry to be exactly pi
        self.phase_shifts[-1] = -1 * np.pi

        #TODO: remove for our dataset
        self.phase_shifts = self.phase_shifts * -1 #just for deepsense dataset

        self.angle_bins = np.arcsin(self.phase_shifts / pi)
        
        #define the angle bins to plot
        self.angle_bins_to_keep = (self.angle_bins > self.radar_fov[0]) & (self.angle_bins < self.radar_fov[1])
        
        #mesh grid coordinates for plotting
        self.thetas,self.rhos = np.meshgrid(self.angle_bins[self.angle_bins_to_keep],self.range_bins[:self.max_range_bin])
        self.x_s = np.multiply(self.rhos,np.sin(self.thetas))
        self.y_s = np.multiply(self.rhos,np.cos(self.thetas))

    def plot_range_azimuth_response(
            self,
            sample_idx:int, 
            ax_cartesian = None,
            ax_spherical = None,
            show = True):
        """Plot the range-azimuth response in cartesian and spherical coordinates

        Args:
            sample_idx (int): The sample index,
            ax_cartesian (Axes): axes to plot the cartesian plot on. Defaults to None
            ax_spherical (Axes): axes to plot the spherical plot on. Defaults to None
            show (bool): on True shows the plot. Defaults to True
        """

        #setup the axes
        if (ax_cartesian == None) or (ax_spherical == None):

            fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
            fig.subplots_adjust(wspace=0.2)

            ax_cartesian = axs[0]
            ax_spherical = axs[1]

        #get the raw ADC data cube
        adc_data_cube = self._get_raw_ADC_data_cube(sample_idx + self.num_previous_frames)

        #compute the frame range-azimuth response
        range_azimuth_response = self._compute_frame_normalized_range_azimuth_heatmaps(adc_data_cube)
        
        #plot the response in cartesian for the first chirp
        self._plot_range_azimuth_heatmap_cartesian(range_azimuth_response[:,:,0],
                                                   ax=ax_cartesian,
                                                   show=False)
        
        #plot the response in spherical coordinates
        self._plot_range_azimuth_heatmap_spherical(range_azimuth_response[:,:,0],
                                                   ax=ax_spherical,
                                                   show=False)
        
        if show:
            plt.show()
        return
    
    def plot_from_saved_range_azimuth_response(
            self,
            sample_idx:int, 
            ax_cartesian = None,
            ax_spherical = None,
            show = True):
        """Plot the range-azimuth response in cartesian and spherical coordinates
        from a previously saved response

        Args:
            sample_idx (int): The sample index,
            ax_cartesian (Axes): axes to plot the cartesian plot on. Defaults to None
            ax_spherical (Axes): axes to plot the spherical plot on. Defaults to None
            show (bool): on True shows the plot. Defaults to True
        """
        
        #set up the axes
        if (ax_cartesian == None) or (ax_spherical == None):

            fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
            fig.subplots_adjust(wspace=0.2)

            ax_cartesian = axs[0]
            ax_spherical = axs[1]

        range_azimuth_response = self.load_range_az_spherical_from_file(sample_idx=sample_idx)

        #account for the potential of multiple frames and chirps being plotted
        if not self.use_average_range_az:
            idx_to_plot = self.num_previous_frames * self.num_chirps_to_save
        else:
            idx_to_plot = self.num_previous_frames
        
        #plot the response in cartesian for the first chirp
        self._plot_range_azimuth_heatmap_cartesian(range_azimuth_response[:,:,idx_to_plot],
                                                   ax=ax_cartesian,
                                                   show=False)
        
        #plot the response in spherical coordinates
        self._plot_range_azimuth_heatmap_spherical(range_azimuth_response[:,:,idx_to_plot],
                                                   ax=ax_spherical,
                                                   show=False)
        
        if show:
            plt.show()
    
    def generate_and_save_range_azimuth_response(self,sample_idx:int):
        """Compute the range_azimuth response and save it to a file

        Args:
            sample_idx (int): The sample index for which to generate and save the result to
        """

        if self.num_previous_frames == 0:
            #get the raw ADC data cube
            adc_data_cube = self._get_raw_ADC_data_cube(sample_idx)

            #compute the frame range-azimuth response
            range_azimuth_response = self._compute_frame_normalized_range_azimuth_heatmaps(adc_data_cube)

            #save the generated response to a file
            self._save_range_az_spherical_to_file(range_azimuth_response,sample_idx=sample_idx)

        else:

            if self.use_average_range_az:
                range_azimuth_response = np.zeros((self.max_range_bin,
                                                np.sum(self.angle_bins_to_keep),
                                                (self.num_previous_frames + 1)))

                for i in range(self.num_previous_frames + 1):
                    #get the raw ADC data cube
                    adc_data_cube = self._get_raw_ADC_data_cube(
                        sample_idx + i)

                    #compute the frame range-azimuth response
                    range_azimuth_response[:,:,i] = \
                        self._compute_frame_normalized_range_azimuth_heatmaps(adc_data_cube)


                self._save_range_az_spherical_to_file(range_azimuth_response,sample_idx=sample_idx)

            else:
                range_azimuth_response = np.zeros((self.max_range_bin,
                                                np.sum(self.angle_bins_to_keep),
                                                self.num_chirps_to_save * (self.num_previous_frames + 1)))

                for i in range(self.num_previous_frames + 1):
                    #get the raw ADC data cube
                    adc_data_cube = self._get_raw_ADC_data_cube(
                        sample_idx + i)

                    #compute the frame range-azimuth response
                    start_idx = i * self.num_chirps_to_save
                    stop_idx = start_idx + self.num_chirps_to_save
                    range_azimuth_response[:,:,start_idx:stop_idx] = \
                        self._compute_frame_normalized_range_azimuth_heatmaps(adc_data_cube)


                self._save_range_az_spherical_to_file(range_azimuth_response,sample_idx=sample_idx)
        return
    
    def generate_and_save_all_grids(self, clear_contents=False):
        """Save all of the loaded radar point clouds to files

        Args:
            clear_contents(bool,optional): on True, will clear the previously generated dataset, Defaults to False
        """

        self._check_save_directory(clear_contents)
        
        num_files = len(self.radar_data_paths)

        for i in tqdm(range(num_files - self.num_previous_frames)):
            self.generate_and_save_range_azimuth_response(sample_idx=i)

# HElper Functions

    #checking the save directory

    def _check_save_directory(self,clear_contents = False):

        path = self.save_file_folder

        if os.path.isdir(path):
            print("RadarDataProcessor._check_save_directory: found directory {}".format(path))

            if clear_contents:
                print("RadarDataProcessor._check_save_directory: clearing contents of {}".format(path))

                #clear the contents
                for file in os.listdir(path):
                    file_path = os.path.join(path,file)

                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print("Failed to delete {}".format(path))
            else:

                contents = sorted(os.listdir(path))

                if len(contents) > 0:
                    last_file = contents[-1]

                    #get the most recent sample number
                    sample_number = last_file.split("_")[1]
                    sample_number = int(sample_number.split(".")[0]) \
                        - self.save_file_number_offset
                
                    #set the offset for saving files
                    self.save_file_start_offset = sample_number
                    print("RadarDataProcessor._check_save_directory: detected existing samples, starting on sample {}".format(self.save_file_start_offset))
                    
        else:
            print("RadarDataProcessor._check_save_directory: creating directory {}".format(path))
            os.makedirs(path)


    # def load_data_from_DCA1000(self,file_path):
        
    #     #TODO: Need to update this function to support loading data in from the DCA1000
    #     #import the raw data
    #     LVDS_lanes = 4
    #     adc_data = np.fromfile(file_path,dtype=np.int16)

    #     #reshape to get the real and imaginary parts
    #     adc_data = np.reshape(adc_data, (LVDS_lanes * 2,-1),order= "F")

    #     #convert into a complex format
    #     adc_data = adc_data[0:4,:] + 1j * adc_data[4:,:]

    #     #reshape to index as [rx channel, sample, chirp, frame]
    #     adc_data_cube = np.reshape(adc_data,(self.rx_channels,self.samples_per_chirp,self.chirp_loops_per_frame,-1),order="F")

    def _get_raw_ADC_data_cube(self,sample_idx:int):
        """Get the raw ADC data cube associated with the given data sample

        Args:
            sample_idx (int): the sample index to get the adc data cube for

        Returns:
            np.ndarray: the adc data cube indexed by (indexed by [rx channel, sample, chirp])
        """

        path = self.radar_data_paths[sample_idx]

        if ".npy" in path:
            return np.load(path)
        elif ".mat" in path:
            return loadmat(path)['data']

        return np.load(path)
    
    def _compute_frame_normalized_range_azimuth_heatmaps(self,adc_data_cube:np.ndarray):

        frame_range_az_heatmaps = np.zeros((
            self.max_range_bin,
            np.sum(self.angle_bins_to_keep),
            self.num_chirps_to_save))

        for i in range(self.num_chirps_to_save):
            frame_range_az_heatmaps[:,:,i] = self._compute_chirp_normalized_range_azimuth_heatmap(adc_data_cube,chirp=i)
    
        if self.use_average_range_az:
            frame_range_az_heatmaps = np.average(frame_range_az_heatmaps, axis=2)

            #add an extra dimmension
            frame_range_az_heatmaps = frame_range_az_heatmaps[...,np.newaxis]
        
        return frame_range_az_heatmaps
    
    def _compute_chirp_normalized_range_azimuth_heatmap(self,adc_data_cube:np.ndarray,chirp=0):
        """Compute the range azimuth heatmap for a single chirp in the raw ADC data frame

        Args:
            adc_data_cube (np.ndarray): _description_
            chirp (int, optional): _description_. Defaults to 0.

        Returns:
            np.ndarray: the computed range-azimuth heatmap (normalized and thresholded)
                NOTE: the output is cropped for only the desired ranges and angles
        """

        #get range angle cube
        data = np.zeros((self.samples_per_chirp,self.num_angle_bins),dtype=complex)
        data[:,0:self.rx_channels] = np.transpose(adc_data_cube[:,:,chirp])

        #compute Range FFT
        data = np.fft.fftshift(np.fft.fft(data,axis=0))

        #compute azimuth response
        data = 20* np.log10(np.abs(np.fft.fftshift(np.fft.fft(data,axis=-1))))

        #[for debugging] to get an idea of what the max should be
        max_db = np.max(data)
        
        #filter to only output the desired ranges and angles
        data = data[:self.max_range_bin,self.angle_bins_to_keep]

        #perform thresholding on the input data
        data[data <= self.power_range_dB[0]] = self.power_range_dB[0]
        data[data >= self.power_range_dB[1]] = self.power_range_dB[1]
        
        #normalize the data
        data = (data - self.power_range_dB[0]) / \
            (self.power_range_dB[1] - self.power_range_dB[0])

        return data
    
    def _plot_range_azimuth_heatmap_cartesian(self,
                                              rng_az_response:np.ndarray,
                                              ax:plt.Axes=None,
                                              show=True):
        """Plot the range azimuth heatmap (for a single chirp) in cartesian coordinates

        Args:
            rng_az_response (np.ndarray): num_range_bins x num_angle_bins normalized range azimuth response
            ax (plt.Axes, optional): The axis to plot on. If none provided, one is created. Defaults to None.
            show (bool): on True, shows plot. Default to True
        """
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot()
        
        cartesian_plot = ax.pcolormesh(
            self.x_s,
            self.y_s,
            rng_az_response,
            shading='gouraud',
            cmap="gray")
        ax.set_xlabel('X (m)',fontsize=RadarDataProcessor.font_size_axis_labels)
        ax.set_ylabel('Y (m)',fontsize=RadarDataProcessor.font_size_axis_labels)
        ax.set_title('Range-Azimuth\nHeatmap (Cart.)',fontsize=RadarDataProcessor.font_size_title)
        ax.tick_params(labelsize=RadarDataProcessor.font_size_ticks)

        #set the line width
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(RadarDataProcessor.line_width_axis) # change width


        if show:
            plt.show()
        #if enable_color_bar:
        #    cbar = self.fig.colorbar(cartesian_plot)
        #    cbar.set_label("Relative Power (dB)",size=RadarDataProcessor.font_size_color_bar)
        #    cbar.ax.tick_params(labelsize=RadarDataProcessor.font_size_color_bar)
    
    def _plot_range_azimuth_heatmap_spherical(self,
                                              rng_az_response:np.ndarray,
                                              ax:plt.Axes = None,
                                              show = True):
        """Plot the range azimuth heatmap in spherical coordinates

        Args:
            rng_az_response (np.ndarray): num_range_bins x num_angle_bins normalized range azimuth response
            ax (plt.Axes, optional): The axis to plot on. If none provided, one is created. Defaults to None.
            show (bool): on True, shows plot. Default to True
        """

        if not ax:
            fig = plt.fig()
            ax = fig.add_subplot()

        #plot polar coordinates
        max_range = self.max_range_bin * self.range_res
        min_angle = min(self.angle_bins[self.angle_bins_to_keep])
        max_angle = max(self.angle_bins[self.angle_bins_to_keep])
        ax.imshow(np.flip(rng_az_response,axis=0),
                  cmap="gray",
                  extent=[max_angle,min_angle,
                          self.range_bins[0],max_range],
                          aspect='auto')
        ax.set_xlabel('Angle(radians)',fontsize=RadarDataProcessor.font_size_axis_labels)
        ax.set_ylabel('Range (m)',fontsize=RadarDataProcessor.font_size_axis_labels)
        ax.set_title('Range-Azimuth\nHeatmap (Polar)',fontsize=RadarDataProcessor.font_size_title)
        ax.tick_params(labelsize=RadarDataProcessor.font_size_ticks)

        #set the line width
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(RadarDataProcessor.line_width_axis) # change width

        #if enable_color_bar:
        #    cbar = self.fig.colorbar(polar_plt)
        #    cbar.set_label("Relative Power (dB)",size=RadarDataProcessor.font_size_color_bar)
        #    cbar.ax.tick_params(labelsize=RadarDataProcessor.font_size_color_bar)
        if show:
            plt.show()

#save to a file

    def _save_range_az_spherical_to_file(self,range_azimuth_response:np.ndarray,sample_idx:int):
        """Save the given range-azimuth response (in spherical) to a file at the configured location

        Args:
            range_azimuth_response (np.ndarray): The range azimuth response to save
            sample_idx (int): The index of the sample to be saved
        """

        #determine the full path and file name
        file_name = "{}_{}.npy".format(
            self.save_file_name,
            sample_idx + self.save_file_number_offset + self.save_file_start_offset)
        path = os.path.join(self.save_file_folder,file_name)

        #save the file to a .npy array
        np.save(path,range_azimuth_response)

        return
    
    def load_range_az_spherical_from_file(self,sample_idx:int):
        """Load a previously saved range-azimuth response from a file

        Args:
            sample_idx (int): The sample index of the file to load

        Returns:
            np.ndarray: The loaded range-azimuth response
        """

        #determine the full path and file name
        file_name = "{}_{}.npy".format(
            self.save_file_name,
            sample_idx + self.save_file_number_offset)
        path = os.path.join(self.save_file_folder,file_name)

        #load the grid
        return np.load(path)


