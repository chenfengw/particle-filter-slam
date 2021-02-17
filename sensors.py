import numpy as np
import pandas as pd
from pr2_utils import read_data_from_csv

class Sensor:
    def __init__(self,path_to_data):
        self.data_path = path_to_data
        self.timestamp, self.data = read_data_from_csv(self.data_path)
        # update rate in second
        self.delta_t = np.diff(self.timestamp).mean() / 1e9

    def get_time(self):
        return self.timestamp

    def get_data(self):
        return self.data

    def get_length(self):
        return len(self.timestamp)

class Gyroscope(Sensor):
    def __init__(self,path_to_data):
        super().__init__(path_to_data)
        self.delta_yaw = self.data[:,-1]
        self.angular_velocity = self.get_angular_velocity()
 
    def get_angular_velocity(self):
        return self.delta_yaw / self.delta_t
    

class Encoder(Sensor):
    def __init__(self,path_to_data):
        super().__init__(path_to_data)
        self.left_count = self.data[:,0] 
        self.right_count = self.data[:,1] 
        self.diameter = (0.623479 + 0.622806) /2
        self.resolution = 4096
        self.meter_per_tick = np.pi * self.diameter / self.resolution

        # calculate speed for each wheel
        self.speed_l = self.get_linear_speed(self.left_count)
        self.speed_r = self.get_linear_speed(self.right_count)
        self.linear_velocity = (self.speed_l + self.speed_l) / 2
        self.timestamp = self.timestamp[1:]

    def get_linear_speed(self, tick_counts):
        n_ticks = np.diff(tick_counts)
        return n_ticks * self.meter_per_tick / self.delta_t

class Lidar(Sensor):
    def __init__(self,path_to_data):
        super().__init__(path_to_data)

    def polar_to_xy(self, row_index, max_range=80, min_range=0.1):
        """convert lidar data to xy coordinates in sensor frame

        Args:
            lidar_data (np array): entire lidar data array
            row_index (int): index of the row. Row represents timestamp
            max_range (int, optional): max range of lidar. Defaults to 80.
            min_range (float, optional): min range of lidar. Defaults to 0.1.

        Returns:
            np array: 2 x n_samples, first row x, second row y
        """
        angles = np.linspace(-5, 185, 286) / 180 * np.pi
        ranges = self.data[row_index, :]
        
        # take valid indices
        indValid = np.logical_and((ranges < max_range), (ranges > min_range))
        ranges = ranges[indValid]
        angles = angles[indValid]

        # xy position in the sensor frame
        xs0 = ranges*np.cos(angles)
        ys0 = ranges*np.sin(angles)
        
        # convert position in the map frame here 
        return np.stack((xs0,ys0))

class StereoCamera(Sensor):
    def __init__(self,path_to_data):
        super().__init__(path_to_data)