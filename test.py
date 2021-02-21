#%% 
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import pr2_utils as slam_utils
import importlib
from pr2_utils import *
import sensors
importlib.reload(sensors)
# %%
all_sensors = {}
all_sensors["gyro"] = sensors.Gyroscope('data/sensor_data/fog.csv')
all_sensors["lidar"] = sensors.Lidar('data/sensor_data/lidar.csv')
all_sensors["encoder"] = sensors.Encoder('data/sensor_data/encoder.csv')
# %% test mapCorrelation
lidar_data = all_sensors["lidar"].get_data()
angles = np.linspace(-5, 185, 286) / 180 * np.pi
ranges = lidar_data[0, :]

# take valid indices
indValid = np.logical_and((ranges < 80),(ranges> 0.1))
ranges = ranges[indValid]
angles = angles[indValid]

# init MAP
MAP = {}
MAP['res']   = 0.1 #meters
MAP['xmin']  = -50  #meters
MAP['ymin']  = -50 # in meter
MAP['xmax']  =  50 # in meter
MAP['ymax']  =  50 # in meter
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8

#import pdb
#pdb.set_trace()

# xy position in the sensor frame
xs0 = ranges*np.cos(angles)
ys0 = ranges*np.sin(angles)

# convert position in the map frame here 
Y = np.stack((xs0,ys0))

# convert from meters to cells
xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

# build an arbitrary map 
# indGood are index where laser scan hit
indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
MAP['map'][xis[indGood],yis[indGood]]=1

# %%
# x coordinates in m of all pixel(cell) in map
x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
# y coordinates in m of all pixel (cell)
y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map

x_range = np.arange(-0.4,0.4+0.1,0.1) # in meter
y_range = np.arange(-0.4,0.4+0.1,0.1)

print("x_range length: {}, y_range length {}".format(len(x_range),len(y_range)))
# %%
print("Testing map_correlation with {}x{} cells".format(MAP['sizex'],MAP['sizey']))
ts = tic()
c = mapCorrelation(MAP['map'],x_im,y_im,Y,x_range,y_range) # Y is the laser xy coordinates
print(c)
print(c.shape)
toc(ts,"Map Correlation")

# %%
c_ex = np.array([[ 4.,  6.,  6.,  5.,  8.,  6.,  3.,  2.,  0.],
                [ 7.,  5., 11.,  8.,  5.,  8.,  5.,  4.,  2.],
                [ 5.,  7., 11.,  8., 12.,  5.,  2.,  1.,  5.],
                [ 6.,  8., 13., 66., 33.,  4.,  3.,  3.,  0.],
                [ 5.,  9.,  9., 63., 55., 13.,  5.,  7.,  4.],
                [ 1.,  1., 11., 15., 12., 13.,  6., 10.,  7.],
                [ 2.,  5.,  7., 11.,  7.,  8.,  8.,  6.,  4.],
                [ 3.,  6.,  9.,  8.,  7.,  7.,  4.,  4.,  3.],
                [ 2.,  3.,  2.,  6.,  8.,  4.,  5.,  5.,  0.]])

if np.sum(c==c_ex) == np.size(c_ex):
    print("...Test passed.")
else:
    print("...Test failed. Close figures to continue tests.")	

# %%
#plot original lidar points
fig1 = plt.figure()
plt.plot(xs0,ys0,'.k')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Laser reading")
plt.axis('equal')

#plot map
fig2 = plt.figure()
plt.imshow(MAP['map'],cmap="hot");
plt.title("Occupancy grid map")

#plot correlation
fig3 = plt.figure()
ax3 = fig3.gca(projection='3d')
X, Y = np.meshgrid(np.arange(0,9), np.arange(0,9))
ax3.plot_surface(X,Y,c,linewidth=0,cmap=plt.cm.jet, antialiased=False,rstride=1, cstride=1)
plt.title("Correlation coefficient map")

# %%
import math
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
# %%
import time
now = time.time()

x = np.random.rand(11586500)
for i in range(2):
    for i in range(11586500):
        y = np.random.normal()
        find_nearest(x,y)
print(time.time() - now)
# %%
