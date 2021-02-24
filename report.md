# ECE 276A Project 2 Report

## Particle Filter SLAM

## Introduction

Simultaneous localization and mapping has very wide application in robotics, autonomous driving, and many other applications. Robot needs to where it is in relation with the world and what the world looks like. However this is a chicken and egg problem: to localize robot we need a map, and to construct a map we need to know robot's location. SLAM solves this problem by performing localization and mapping at the same time. In this project we will implement particle filter based SLAM algorithm. 

## Problem Formulation
Let $p(\textbf{x}_{0:T}, \textbf{m} | \textbf{z}_{0:T}, \textbf{u}_{0:T-1})$ be the probability density function of robot trajectory and map given sensor measurements and control inputs.

_**Problem:**_ Find the best estimator $\textbf{x}_{0:T}, \textbf{m}$ that maximize $p(\textbf{x}_{0:T}, \textbf{m} | \textbf{z}_{0:T}, \textbf{u}_{0:T-1})$ . 

## Technical Approach

### Particle Filter

Particle filter works by proposing pose estimates based on robot's motion model and use sensor measurements to filter poor quality estimations and keep the good ones. Hence allowing robot to be localized with high degree of accuracy. Each particle contains a pose hypothesis $\mu$ and corresponding probability $\alpha$. We initialize particle filter with 25 particles at the origin. 

### Predict
We use differential drive model to predict the post of the robot. This model calculates car's location $\textbf{p} \in \R^3$ and yaw angle $\theta \in [0,2\pi)$ using  angular velocity and linear velocity. We can obtain angular velocity from the gyroscope, which measures $\Delta \theta$ between consecutive timestamps, and linear velocity from encoder which counts wheel rotations. The car's state $\textbf{x} = (\textbf{p}, \theta)$ at $t+1$ can be obtained as follows:
$$
\textbf{x}_{t+1} = \begin{bmatrix} x_{t+1} \\ y_{t+1} \\ \theta_{t+1} \end{bmatrix} = \textbf{x}_{t} + \tau \begin{bmatrix} v_t \cos(\theta_t) \\ v_t \sin(\theta_t) \\ \omega_{t} \end{bmatrix}
$$

#### Calculating Angular Velocity 
The gyro data provides change in angles $\Delta \theta = \tau \omega$  between two consecutive timestamps. This means if we calculate the time difference $\tau$ between two conceptive reporting we can get angular velocity using the formula $\omega = \frac{\Delta \theta}{\tau}$.

#### Calculating Linear Velocity 
The encoder data provides accumulated encoder counts since the beginning of the time.  To calculate linear velocity, we first measure the distance traveled between two consecutive timestamps and then divide by the time it took to travel that distance. This procedure can  be summarized by the formula below:
$$
v = \frac{\pi dz}{4096\tau}
$$
$d$: diameter of the wheel

$z$: number of counts between two consecutive timestamp

$\tau$: time elapsed between two consecutive measurements 

#### Noise
Since our sensor measurements is not perfect, we add gaussian noise $\epsilon \sim \mathcal{N}(0,\sigma)$ to  the data.
$$
\begin{align*} 
v &=  v + \epsilon_{v}  & \epsilon_{v} &\sim \mathcal{N}(0,\sigma_{v})\\ 
\omega &=\omega + \epsilon_{\omega} & \epsilon_{\omega} &\sim \mathcal{N}(0,\sigma_{\omega})
\end{align*}
$$
### Update

During prediction step, all $\mu$ are updated using the motion model. Now we can update $\alpha$ based on lidar scan and see which particle has the highest probability being the true pose. The weight of particle $\alpha$ is proportional to $\exp f(\mathbf{y},\mathbf{m})$ where
$$
\begin{align*} 
\alpha &\propto \exp f(\mathbf{y},\mathbf{m}) \\
\\
f(\mathbf{y},\mathbf{m}) &= \sum_{y_i}\mathbf{1\{y_i = m_i\}} \\
\end{align*}
$$
$f(\mathbf{y},\mathbf{m})$ is the laser correlation model. It takes lidar scan $\mathbf{y}$ in the world frame, the map $\mathbf{m}$, and produces a correlation number indicating how likely is $\mu$ being the true pose. To compute laser coronation, we first convert lidar scan from vehicle frame to world frame using particle pose $\mu$ then feed $\mathbf{y}$ and $\mathbf{m}$ into $f(\mathbf{y},\mathbf{m})$ to get laser correlation value for each particle. Finally we pass correlation values for all particles into softmax function to make sure  $\sum_{i=1}^{K}\alpha_{i} = 1$.

### Occupancy Grid Mapping

In order to create a map, we employee occupancy grid mapping. The entire map is discretized into cells of fixed size. Each cell is in one of the three possible states: free, occupied, and unexplored. To update the map, we need to know where the car is. This information can be obtained by taking the particle associated with the largest weight as our best estimator of car's pose. Using this pose information, we can transform lidar scan from vehicle frame to world frame and use `skimage.draw.line` algorithm to determine cells that laser beams pass through. Laser beams pass through free space and terminate at obstacles, so we mark the cell at the end of the line to be occupied and the rest of the cells on the line to be free.  During each iteration, free cells are decremented by $\log4$, occupied cells are incremented by $\log4$, and unexplored cells remain unchanged.
$$
\text{cell value} = 
\begin{cases}
+ \log4 & \text{if cell is occupied} \\
- \log4 & \text{if cell is free} 
\end{cases}
$$
### Texture Mapping



## Results
### Occupancy Grid

Below is the occupancy grid for every 100,000 loop.


<img src="/Users/Charlie/Documents/Work_School/UCSD/Grad/Winter_2021/ECE276A/project/project2/figs/map_progress.svg" alt="map_progress" style="zoom:70%;" title="eee" />



### Car Trajectory

Below is the car trajectory for every 100,000 loop. Data is downsampled 100 times for display.

<img src="/Users/Charlie/Documents/Work_School/UCSD/Grad/Winter_2021/ECE276A/project/project2/figs/trajactory_all.svg" alt="trajactory_all" style="zoom:70%;" />

### Parameters
| Name     | Value       |
| :-------------: |:-------------:|
| Map Resolution | 1 m / pixel |
| Map Size | x: [-1300 m, 1300 m], y: [-1200 m, 1200 m] |
|                |                                            |


### Conclusion

Particle filter worked incredibly well for simultaneous localization and mapping, generated accurate car trajectory and highly detailed map for city street. Unfortunately our implementation is quite slow, 25 particles will take about 3hr to run. We think the speed can be dramatically improved if we use `cv2.findContours`  instead of `skimage.draw.line`. Since `findContours` can rasterize all rays at the same time whereas `skimage.draw.line` can only rasterize one ray at the time. By processing line rasterization in parallel, `findContours` can reduce number of `for` loop in the code and speed up the map update. 

In addition, we found speed can also be improved if we just use 1 particle instead of 25. Single particle only takes around 25min to run, and it actually yields surprisingly good results as all plots above are generated using single particle. We think this is due to highly accurate measurements from fiber optic gyro. In fact, FOG is regraded as state of the art gyroscope technology as it's extremely accurate and highly reliable. Therefore we conclude particle filter with high particle number is only beneficial when sensor data is noisy. If sensor data is very accurate, a small number of particles can produce efficient and accurate results. 















