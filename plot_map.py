# %%
import numpy as np
import matplotlib.pyplot as plt
from occupancy_map import Map
# %% load map and car car_trajactory
map_all = np.load("map_progress.npy")
trajactory_all = np.load("traj_progress.npy")
map_final = np.load("data_processed/map_log_odds.npy")
trajactory_final = np.load("data_processed/car_trajactory.npy")

indexes = [1,2,3,5,6,7,10,11]
map_all2 = [item for item in map_all[indexes]]
map_all2.append(map_final)

traj_all = [item for item in trajactory_all[indexes]]
traj_all.append(trajactory_final)
# %% plot map
plt.figure(figsize=[15,15])
for idx in range(len(map_all2)):
    map_raw = map_all2[idx]
    map_display = Map().render_map(map_raw)
    
    plt.subplot(3,3,idx+1)
    plt.imshow(map_display[1000:,1000:],cmap="gray",vmin=0, vmax=1)
plt.savefig("map_progress.svg",bbox_inches="tight")

# %% plot trajactory_all
plt.figure(figsize=[15,13])
for idx in range(len(traj_all)):
    traj = traj_all[idx]
    
    plt.subplot(3,3,idx+1)
    plt.scatter(traj[0][::1000],traj[1][::1000])
plt.savefig("trajactory_all.svg",bbox_inches="tight")
# %%
