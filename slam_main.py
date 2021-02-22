#%% 
import numpy as np
import matplotlib.pyplot as plt
import time
import importlib
from skimage.draw import line

import sensors
import occupancy_map
import particle_filter
import transform
importlib.reload(sensors)
importlib.reload(occupancy_map)
importlib.reload(particle_filter)
importlib.reload(transform)

from occupancy_map import Map
from transform import Transform
from particle_filter import ParticleFilter
# %% import sensor
lidar = sensors.Lidar('data/sensor_data/lidar.csv',downsample_rate=10)
gyro = sensors.Gyroscope('data/sensor_data/fog.csv',downsample_rate=100)
encoder = sensors.Encoder('data/sensor_data/encoder.csv',downsample_rate=10)

# %% test timestamp
plt.figure()
plt.plot(lidar.timestamp)
plt.plot(gyro.timestamp)
plt.plot(encoder.timestamp)
# %% initialization
# myMap = Map(res=1,x_range=[-1300,1300],y_range=[-1200,1200])
myMap = Map(res=1,x_range=[-150,150],y_range=[-150,150])
tf = Transform()
pf = ParticleFilter(n_particles=1,)
# %% set first lidar scan
lidar1_scan = lidar.polar_to_xy(0)
lidar_world = tf.lidar_to_world(lidar1_scan, np.zeros(3))
x_cells, y_cells = myMap.meter_to_cell(lidar_world[:2,:])

scaned_cell = []
for x_cell1, y_cell1 in zip(x_cells,y_cells):
    x_cell, y_cell = line(0,0,x_cell1,y_cell1)
    scaned_cell.append((x_cell,y_cell))

myMap.update_log_odds(scaned_cell)
plt.imshow(myMap.map,cmap="gray")

# %% Main Loop
gyro_idx = int(-1)
gyro_times = gyro.timestamp[:gyro_idx]
gyro_data = gyro.angular_velocity[:gyro_idx]
encoder_idx = 0
now = time.time()

for idx, (gyro_time, angular_v) in enumerate(zip(gyro_times,gyro_data)):
    t_loop = time.time()

    encoder_time = encoder.timestamp[encoder_idx]
    lienar_v = encoder.linear_velocity[idx]
    angular_v = gyro.angular_velocity[idx]
    lidar_scan = lidar.polar_to_xy(idx)

    # predict
    pf.predict_all(lienar_v,angular_v,gyro.delta_t)
    
    # update alpha
    max_idx, corrlation_max = pf.update(lidar_scan, myMap, tf)

    # pick max alpha and update map
    particle_max = pf.particles[:,max_idx]
    lidar_world = tf.lidar_to_world(lidar_scan, particle_max)
    x_cells, y_cells = myMap.meter_to_cell(lidar_world[:2,:])

    # find cells that ray path throw
    scaned_cell = []
    particle_x = np.round(particle_max[0] / myMap.res).astype(int)
    particle_y = np.round(particle_max[1] / myMap.res).astype(int)
    for x_cell1, y_cell1 in zip(x_cells,y_cells):
        x_cell, y_cell = line(particle_x, particle_y, x_cell1, y_cell1)
        scaned_cell.append((x_cell,y_cell))

    myMap.update_log_odds(scaned_cell)

    # determine if need resampling
    pf.resampling()

    print("loop idx: {}, this loop took: {}".format(idx, time.time()- t_loop))



# %%
myMap.show_map()
# %%
