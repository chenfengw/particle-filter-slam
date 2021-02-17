#%% 
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import pr2_utils as slam_utils
import importlib

import sensors
importlib.reload(sensors)
# %%
all_sensors = {}
all_sensors["gyro"] = sensors.Gyroscope('data/sensor_data/fog.csv')
all_sensors["lidar"] = sensors.Lidar('data/sensor_data/lidar.csv')
all_sensors["encoder"] = sensors.Encoder('data/sensor_data/encoder.csv')
# %%
for key, val in all_sensors.items():
    print("{}, length is {}".format(key,val.get_length()))

    plt.plot(val.get_time(),label = key)
plt.legend()
plt.xlabel("sample count")
plt.ylabel("time stamp")
plt.title("time vs sample count")
plt.savefig("figs/time vs sample count.png",bbox_inches="tight")
# %%
plt.plot(all_sensors["encoder"].timestamp[-10:])
plt.plot(all_sensors["lidar"].timestamp[-10:])
# %%
plt.plot(all_sensors["encoder"].timestamp[:500] - all_sensors["lidar"].timestamp[:500]/1e9)
# %%

# %% test ParticleFilter
import particle_filter 
importlib.reload(particle_filter)
pf = particle_filter.ParticleFilter()
gyro = all_sensors["gyro"] 
encoder = all_sensors["encoder"]

# %%
x = np.zeros(3)
traj_all = []
traj_all.append(x)

for encoder_idx in range(encoder.get_length()-1):
    linear_v = encoder.linear_velocity[encoder_idx]
    t1 = encoder.timestamp[encoder_idx]
    t2 = encoder.timestamp[encoder_idx+1]
    gyro_match = gyro.angular_velocity[(gyro.timestamp >= t1) & (gyro.timestamp < t2)]

    for omega in gyro_match:
        x = pf.predict(x,linear_v,omega,gyro.delta_t)
        traj_all.append(x)
# %%
x_all = [item[0] for item in traj_all]
y_all = [item[1] for item in traj_all]
plt.plot(x_all,y_all)
# %%
