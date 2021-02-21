import numpy as np
import pr2_utils 
import matplotlib.pyplot as plt

class Map:
    def __init__(self, res=0.1, x_range=[-50, 50], y_range=[-50, 50]):
        self.res = 0.1  # 0.1 meter per pixel
        self.xmin = x_range[0]
        self.xmax = x_range[1]
        self.ymin = y_range[0]
        self.ymax = y_range[1]
        self.sizex = int(np.ceil((self.xmax - self.xmin) / self.res + 1))
        self.sizey = int(np.ceil((self.ymax - self.ymin) / self.res + 1))
        self.map = np.zeros([self.sizex, self.sizey])

        # calculation needed for map correlation
        # x_in_meter: x coordinates in m of all pixel(cell) in map
        self.x_in_meter = np.arange(self.xmin, self.xmax+self.res, self.res)
        self.y_in_meter = np.arange(self.ymin, self.ymax+self.res, self.res)

        # search space for map correlation
        self.scan_x_range = np.arange(-0.4, 0.4+0.1, 0.1)  # in meter
        self.scan_y_range = np.arange(-0.4, 0.4+0.1, 0.1)

    def map_correlation(self, lidar_scan):
        return pr2_utils.mapCorrelation(self.map, 
                                        self.x_in_meter, 
                                        self.y_in_meter, 
                                        lidar_scan, 
                                        self.scan_x_range, 
                                        self.scan_y_range)

    def update_log_odds(self):
        pass

    def get_map(self):
        return self.map

    def show_map(self):
        plt.imshow(self.map)