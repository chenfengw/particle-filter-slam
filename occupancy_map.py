import numpy as np
from pr2_utils import mapCorrelation
import matplotlib.pyplot as plt

class Map:
    def __init__(self, res=0.1, x_range=[-50, 50], y_range=[-50, 50]):
        self.res = res  # 0.1 meter per pixel
        self.xmin = x_range[0]
        self.xmax = x_range[1]
        self.ymin = y_range[0]
        self.ymax = y_range[1]
        assert self.xmin < self.xmax
        assert self.ymin < self.ymax

        # get size in pixel
        self._sizex = int(np.ceil((self.xmax - self.xmin) / self.res + 1))
        self._sizey = int(np.ceil((self.ymax - self.ymin) / self.res + 1))
        assert self._sizex % 2 == 1
        assert self._sizey % 2 == 1
        self.map = np.zeros([self._sizex,self._sizey])

        # get center of idx
        self.row_center = (self._sizey - 1) / 2
        self.col_center = (self._sizex - 1) / 2

        # calculation needed for map correlation
        # x_in_meter: x coordinates in m of all pixel(cell) in map
        self.x_in_meter = np.arange(self.xmin, self.xmax+self.res, self.res)
        self.y_in_meter = np.arange(self.ymin, self.ymax+self.res, self.res)

        # search space for map correlation
        self.set_scan_area(depth=4)

        # map for display
        self.map_rendered = None
    def map_correlation(self, lidar_scan):
        return mapCorrelation(self.map > 0, 
                              self.x_in_meter, 
                              self.y_in_meter, 
                              lidar_scan, 
                              self.scan_x_range, 
                              self.scan_y_range)

    def set_scan_area(self, depth=4):
        """update scan area during map correlation

        Args:
            depth (int, optional): search depth. During map correlation,
            the area around particle with depth=n will be searched. Defaults to 4.
            i.e depth=4 means 4 cells to the left/right/up/down around particle will be searched
            for map correlation.
        """
        self.scan_x_range = np.arange(-depth*self.res, depth*self.res + self.res, self.res)  # in meter
        self.scan_y_range = self.scan_x_range

    def meter_to_cell(self,xy_m):
        # xy_m: shape = 2 x n
        # assert xy_m.shape[0] == 2
        x = xy_m[0]
        y = xy_m[1]
        x_cell = np.round(x/self.res)
        y_cell = np.round(y/self.res)

        return x_cell.astype(int), y_cell.astype(int)

    def update_log_odds(self,scacned_cells,odds_ratio=4):
        for x_cell, y_cell in scacned_cells:
            # convert cell to idx
            x_cell = np.int16(x_cell - self.xmin/self.res)
            y_cell = np.int16(y_cell - self.ymin/self.res)
            
            # free cells
            x_free = x_cell[:-1]
            y_free = y_cell[:-1]
            valid1 = self.get_valid_indexes(x_free, y_free)
            self.map[x_free[valid1], y_free[valid1]] -= np.log(odds_ratio)

            # occupied cell
            x_occupied = x_cell[-1]
            y_occupied = y_cell[-1]
            valid2 = self.get_valid_indexes(x_occupied,y_occupied)
            self.map[x_occupied[valid2], y_occupied[valid2]] += np.log(odds_ratio)

    def get_map(self):
        return self.map

    def get_shape(self):
        return self.map.shape

    def get_valid_indexes(self, cellx_idx, celly_idx):
        # get valid indexes for map
        return np.logical_and(np.logical_and(cellx_idx >=0, cellx_idx < self._sizex),
			                  np.logical_and(celly_idx >=0, celly_idx < self._sizey))

    @staticmethod
    def render_map(map_log_odds):
        # set coloring
        map_rendered = np.zeros_like(map_log_odds)
        map_rendered[map_log_odds < 0] = 1 # set free cell to white
        map_rendered[map_log_odds > 0] = 0 # set occupied cell to black
        map_rendered[map_log_odds == 0] = 0.7 # set unexplored cell to gray

        # transpose and flip map
        map_rendered = np.flip(map_rendered.T,axis=0)
        return map_rendered

    def display_map(self):
        if self.map_rendered is None:
            self.map_rendered = self.render_map(self.map)
        plt.imshow(self.map_rendered,cmap="gray",vmin=0, vmax=1)
