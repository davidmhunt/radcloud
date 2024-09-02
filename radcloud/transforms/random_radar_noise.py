import numpy as np

class RandomRadarNoise(object):
    """Adds random noise to the range bins of the radar data
    """

    def __init__(self, noise_level=0.0):

        self.noise_level = noise_level

    def __call__(self, normalized_radar_response:np.ndarray):

        num_range_bins = normalized_radar_response.shape[0]

        for i in range(normalized_radar_response.shape[2]):
            #create a random array of floats to add noise to the range bins
            slice = normalized_radar_response[:,:,i]
            
            noise = self.noise_level * ((2 * np.random.rand(num_range_bins,1) - 1))

            slice = slice + noise

            #fix values that are less than 0 or greater than 1
            slice[slice > 1] = 1.0
            slice[slice < 0] = 0.0

            normalized_radar_response[:,:,i] = slice

        return normalized_radar_response