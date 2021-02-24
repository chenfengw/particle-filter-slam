#%% 
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.draw import line
from tqdm import tqdm_notebook
import cv2
import sensors
from occupancy_map import Map
from transform import Transform
from particle_filter import ParticleFilter
# %% import sensor
lidar = sensors.Lidar('data/sensor_data/lidar.csv',downsample_rate=1)
gyro = sensors.Gyroscope('data/sensor_data/fog.csv',downsample_rate=1)
encoder = sensors.Encoder('data/sensor_data/encoder.csv',downsample_rate=1)

# %% initialization
myMap = Map(res=1,x_range=[-1300,1300],y_range=[-1200,1200])
tf = Transform()

# %% Main Loop
pf = ParticleFilter(n_particles=10, add_noise=True)
gyro_range = int(gyro.get_length() * 1)
now = time.time()
car_trajactory = np.zeros([2,gyro_range])

# initialize index
encoder_idx = 0
update_count = 0
max_idx = min(encoder.get_length()-1, lidar.get_length()-1)
update_now = False

for gyro_idx in tqdm_notebook(range(gyro_range)):
    t_loop = time.time()

    # get time time
    gyro_time = gyro.timestamp[gyro_idx]
    encoder_time = encoder.timestamp[encoder_idx]

    # match timestamp
    if gyro_time > encoder_time and encoder_idx < max_idx:
        encoder_idx += 1
        update_now = True
    
    # predict based on motion model
    linear_v = encoder.linear_velocity[encoder_idx]
    angular_v = gyro.angular_velocity[gyro_idx]
    pf.predict_all(linear_v, angular_v, gyro.delta_t)
    # car_trajactory[:,gyro_idx] = pf.particles[:2,0]

    # update step
    if update_now:
        update_count += 1
        lidar_scan = lidar.polar_to_xy(encoder_idx)

        # update alpha, find max particle
        max_particle, max_correlation = pf.update(lidar_scan, myMap, tf)
        # max_particle = pf.particles[:,0]
        car_trajactory[:,update_count] = max_particle[:2]

        # use max particle to update update map
        lidar_world = tf.lidar_to_world(lidar_scan, max_particle)
        x_cells, y_cells = myMap.meter_to_cell(lidar_world[:2,:])

        # find cells that ray pass through
        scaned_cell = []
        for x_cell1, y_cell1 in zip(x_cells,y_cells):
            particle_x = np.round(pf.particles[0,0] / myMap.res).astype(int)
            particle_y = np.round(pf.particles[1,0] / myMap.res).astype(int)
            x_cell, y_cell = line(particle_x,particle_y,x_cell1,y_cell1)
            scaned_cell.append((x_cell,y_cell))
        myMap.update_log_odds(scaned_cell)
        update_now = False

    pf.resampling()

# print final time
print("total: {} loops, took {} min".format(gyro_idx, (time.time()- now)/60))

# append 0,0 to car_trajactory
origin = np.zeros([2,1])
car_trajactory = np.hstack((origin, car_trajactory))

# %% plot car trajectory
plt.scatter(car_trajactory[0][::100],car_trajactory[1][::100])
# %%
plt.figure()
myMap.display_map()