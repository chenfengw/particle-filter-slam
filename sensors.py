import numpy as np
import pandas as pd
import math
from pr2_utils import read_data_from_csv


class Sensor:
    def __init__(self, path_to_data, downsample_rate=None):
        self.data_path = path_to_data
        self.timestamp, self.data = read_data_from_csv(self.data_path)
        
        if downsample_rate is not None and downsample_rate > 1:
            self.timestamp = self.timestamp[::downsample_rate]
            self.data = self.data[::downsample_rate]

        # update rate in second
        self.delta_t = np.diff(self.timestamp).mean() / 1e9

    def get_time(self):
        return self.timestamp

    def get_data(self):
        return self.data

    def get_length(self):
        return len(self.timestamp)

    def find_idx_at_time(self,given_time,idx_approx):
        if idx_approx <= 500:
            idx_begin = 0
        else:
            idx_begin = idx_approx - 500
        idx_end = idx_begin + 500*2

        if idx_end > self.get_length()-1:
            idx_end = self.get_length() - 1
        
        array_temp = self.timestamp[idx_begin:idx_end]
        return idx_begin + self._find_nearest_idx(array_temp,given_time)

    @staticmethod
    def _find_nearest_idx(array,value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return idx-1
        else:
            return idx

class Gyroscope(Sensor):
    def __init__(self, path_to_data, downsample_rate=None):
        super().__init__(path_to_data, downsample_rate)
        self.delta_yaw = self.data[:, -1]
        self.angular_velocity = self.get_angular_velocity()

    def get_angular_velocity(self):
        return self.delta_yaw / self.delta_t


class Encoder(Sensor):
    def __init__(self, path_to_data, downsample_rate=None):
        super().__init__(path_to_data, downsample_rate)
        self.left_count = self.data[:, 0]
        self.right_count = self.data[:, 1]
        self.diameter = (0.623479 + 0.622806) / 2
        self.resolution = 4096
        self.meter_per_tick = np.pi * self.diameter / self.resolution

        # calculate speed for each wheel
        self.speed_l = self.get_linear_speed(self.left_count)
        self.speed_r = self.get_linear_speed(self.right_count)
        self.linear_velocity = (self.speed_l + self.speed_r) / 2
        self.timestamp = self.timestamp[1:]

    def get_linear_speed(self, tick_counts):
        n_ticks = np.diff(tick_counts)
        return n_ticks * self.meter_per_tick / self.delta_t


class Lidar(Sensor):
    def __init__(self, path_to_data, max_range=80, min_range=0.1, downsample_rate=None):
        super().__init__(path_to_data, downsample_rate)
        self.max_range = max_range
        self.min_range = min_range
        self.angles = np.linspace(-5, 185, 286) / 180 * np.pi #286 rays in radians
    
    def polar_to_xy(self, row_index):
        """convert lidar data to xy coordinates in sensor frame

        Args:
            lidar_data (np array): entire lidar data array
            row_index (int): index of the row. Row represents timestamp

        Returns:
            np array: 4 x n_samples. in homogeneous coordinates. 
            first row x, second row y. Cartesian coordinates in sensor frame.
        """
        ranges = self.data[row_index, :]
        # ranges[ranges==0] = self.max_range  # set 0 to max range

        # take valid indices
        indValid = np.logical_and((ranges < self.max_range), (ranges > self.min_range))
        ranges = ranges[indValid]
        angles = self.angles[indValid]

        # xy position in the sensor frame
        xs0 = ranges*np.cos(angles)
        ys0 = ranges*np.sin(angles)
        ones = np.ones(len(xs0))

        # convert position in the map frame here
        return np.stack((xs0,ys0,ones,ones))


class StereoCamera(Sensor):
    def __init__(self, path_to_data):
        super().__init__(path_to_data)
