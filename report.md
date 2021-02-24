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

### Mapping

In order to create a map, we employee occupancy grid mapping. The entire map is discretized into cells of fixed size. Each cell is in one of the three possible states: free, occupied, and unexplored. To update the map, we need to know where the car is. This information can be obtained by taking the particle associated with the largest weight as our best estimator of car's pose. Using this pose information, we can transform lidar scan from vehicle frame to world frame and use `skimage.draw.line` algorithm to determine cells that laser beams pass through. Laser beams pass through free space and terminate at obstacles, so we mark the cell at the end of the line to be occupied and the rest of the cells on the line to be free.  During each iteration, free cells are decremented by $\log4$, occupied cells are incremented by $\log4$, and unexplored cells remain unchanged.
$$
\text{cell value} = 
\begin{cases}
+ \log4 & \text{if cell is occupied} \\
- \log4 & \text{if cell is free} 
\end{cases}
$$

## Results
### Conclusion

Gaussian classifier worked incredibly well on pixel classification, achieving 100% accuracy on the validation set. We think high accuracy is the result of ample training samples. There are about 1000 images in each class, which enables the classifier to learn the complete distribution of colors. 

Meanwhile, the same classifier design also performs well when segmenting the reclining bin regions. For most of the training and testing image, classifier is able to successfully segment relying bins in the image. The addition of sky blue class helps to reduce false positive samples when sky is in the background. However, this also cause recycling bins whose color are the same as sky to be rejected. 

After segmentation mask is produced, the proposed bounding box detection algorithm works well on the validation set, achieving 100% accuracies on all images. 





















