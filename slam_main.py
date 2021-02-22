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
lidar = sensors.Lidar('data/sensor_data/lidar.csv',downsample_rate=1)
gyro = sensors.Gyroscope('data/sensor_data/fog.csv',downsample_rate=1)
encoder = sensors.Encoder('data/sensor_data/encoder.csv',downsample_rate=1)

# %% test timestamp
plt.figure()
plt.plot(lidar.timestamp)
plt.plot(gyro.timestamp)
plt.plot(encoder.timestamp)
# %% initialization
myMap = Map(res=1,x_range=[-1500,1500],y_range=[-1200,1200])
# myMap = Map(res=1,x_range=[-150,150],y_range=[-150,150])
tf = Transform()

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
pf = ParticleFilter(n_particles=1, add_noise=False)
gyro_range = int(gyro.get_length() * 0.4)
now = time.time()
car_trajactory = np.zeros([2,gyro_range])

# initialize index
encoder_idx = 0
update_count = 0
update_now = False

for gyro_idx in range(gyro_range):
    t_loop = time.time()

    # get time time
    gyro_time = gyro.timestamp[gyro_idx]
    encoder_time = encoder.timestamp[encoder_idx]

    # match timestamp
    if gyro_time > encoder_time and encoder_idx < encoder.get_length()-1:
        encoder_idx += 1
        update_now = True
    
    # predict based on motion model
    linear_v = encoder.linear_velocity[encoder_idx]
    angular_v = gyro.angular_velocity[gyro_idx]
    pf.predict_all(linear_v, angular_v, gyro.delta_t)
    car_trajactory[:,gyro_idx] = pf.particles[:2,0]

    # update step
    if update_now:
        update_count += 1
        lidar_scan = lidar.polar_to_xy(encoder_idx)

        # update alpha, find max alpha
        # max_idx, max_correlation = pf.update(lidar_scan, myMap, tf)

        # use max alpha to update update map
        lidar_world = tf.lidar_to_world(lidar_scan, pf.particles[:,0])
        x_cells, y_cells = myMap.meter_to_cell(lidar_world[:2,:])

        scaned_cell = []
        for x_cell1, y_cell1 in zip(x_cells,y_cells):
            particle_x = np.round(pf.particles[0,0] / myMap.res).astype(int)
            particle_y = np.round(pf.particles[1,0] / myMap.res).astype(int)
            x_cell, y_cell = line(particle_x,particle_y,x_cell1,y_cell1)
            scaned_cell.append((x_cell,y_cell))

        myMap.update_log_odds(scaned_cell)
        update_now = False

    if gyro_idx % 5000 == 0:
        print("loop idx: {}, this loop took: {}".format(gyro_idx, time.time()- t_loop))

# print final time
print("total: {} loops, took {} min".format(gyro_idx, (time.time()- now)/60))

# append 0,0 to car_trajactory
origin = np.zeros([2,1])
car_trajactory = np.hstack((origin, car_trajactory))
# %%
plt.scatter(car_trajactory[0][::1000],car_trajactory[1][::1000])
# plt.savefig("dead_reconking.png",bbox_inches="tight")
# %%
myMap.show_map()
# %%
