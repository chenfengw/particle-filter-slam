# ECE 276A Project 2 Report

## Particle Filter SLAM

## Introduction

Simultaneous localization and mapping has very wide application in robotics, autonomous driving, and many other applications. Robot needs to where it is in relation with the world and what the world looks like. However this is a chicken and egg problem: to localize robot we need a map, and to construct a map we need to know robot's location. SLAM solves this problem by performing localization and mapping at the same time. In this project we will implement particle filter based SLAM algorithm. Particle filter works by proposing pose estimates based on robot's motion model and use sensor measurements to filter poor quality estimations and keep the good ones. Hence allowing robot to be localized with high degree of accuracy. Based on high confident particle location, map is being updated and this procedure repeats again.

## Problem Formulation

### SLAM

Let $p(\textbf{x}_{0:T}, \textbf{m} | \textbf{z}_{0:T}, \textbf{u}_{0:T-1})$ be the probability density function of robot trajectory and map given sensor measurements and control inputs.

_**Problem:**_ Find the best estimator $\textbf{x}_{0:T}, \textbf{m}$ that maximize $p(\textbf{x}_{0:T}, \textbf{m} | \textbf{z}_{0:T}, \textbf{u}_{0:T-1})$ . 



## Technical Approach

### SLAM

#### Motion Model

The motion model we use to model the car is the differential drive model. The model calculates car's location $\textbf{p} \in \R^3$ and yaw angle $\theta \in [0,2\pi)$ using  angular velocity and linear velocity. We can obtain angular velocity from the gyroscope, as it measures $\Delta \theta$ between consecutive timestamps, and linear velocity from encoder which counts wheel rotations. The car's state $\textbf{x} = (\textbf{p}, \theta)$ at $t+1$ can be obtained as follows:
$$
\textbf{x}_{t+1} = \begin{bmatrix} x_{t+1} \\ x_{t+1} \\ \theta_{t+1} \end{bmatrix} = \textbf{x}_{t} + \tau \begin{bmatrix} v_t \cos(\theta_t) \\ v_t \sin(\theta_t) \\ \omega_{t} \end{bmatrix}
$$

#### Calculating Angular Velocity 

The gyro data provides change in angles $\Delta \theta = \tau \omega$  between two consecutive timestamps. This means if we calculate the time difference $\tau$ between two conceptive reporting we can get angular velocity using the formula $\omega = \frac{\Delta \theta}{\tau}$.

#### Calculating Linear Velocity 

The encoder data provides accumulated encoder counts since the beginning of the time. Therefore to calculate the linear velocity, we first measure the distance traveled between two consecutive timestamps and then divide by the time it took to travel that distance, which can be summarized b the formula below:
$$
v = \frac{\pi dz}{4096\tau}
$$
$d$: diameter of the wheel

$z$: number of counts between two consecutive timestamp

$\tau$: time elapsed between two consecutive timestamp

#### Particle Filter
#### Predict

#### Update

#### Mapping 
#### Parameter Estimation

The complete distribution of i.i.d. samples $\mathcal{D}\{x_{i},y_{i}\}_{i=1}^{N}$ can be expressed as:
$$
\begin{align}
p(\mathbf{X},\mathbf{y}) &= \prod_{i=1}^{N}p(x_i|y_i)p(y_i) \\
                                     &= \prod_{i=1}^{N} \prod_{k=1}^{K} \left\{\mathcal{N}(x_i,\mu_{k},\Sigma_{k}) \theta_k \right\}^{\mathbf{1}\{y_i = k\}}                         
\end{align}
$$

where $x_i \in \R^{3}, y_i \in \{1,2,3\dots K\}$

$\mathbf{X} = x_1, x_2,\dots,x_N$ and $\mathbf{y} = y_1,y_2,\dots, y_N$.

Taking the log of joint distribution, we have:
$$
\log p(\mathbf{X},\mathbf{y}) =\sum_{i=1}^{N}\sum_{k=1}^{K} \mathbf{1}\{y_i=k\}(\log \theta_k+ \log \mathcal{N}(x_i,\mu_{k},\Sigma_k)) \\

\begin{align*} 
\theta_k^{*} &=  \underset{\boldsymbol \theta}{\operatorname{argmax}} \log p(\mathbf{X},\mathbf{y})\text{, subject to} \sum_{k=1}^{K} \theta_{k} = 1 \\

\mu_k^{*} &=  \underset{\boldsymbol \mu}{\operatorname{argmax}} \log p(\mathbf{X},\mathbf{y})\\
\Sigma_k^{*} &=  \underset{\boldsymbol \Sigma}{\operatorname{argmax}} \log p(\mathbf{X},\mathbf{y})
\end{align*}
$$
All of the above three optimization problems can be solved by taking the gradient of $\log p(\mathbf{X},\mathbf{y})$ with respect to the variable we are trying to optimize and set the gradient to zero. We can then obtain the following solutions:
$$
\begin{align*} 
\theta_k^{*} &= \frac{1}{N}\sum_{i}^{N} \mathbf{1}\{y_i=k\} \\

\mu_k^{*} &=  \frac{\sum_{i=1}^{N}\mathbf{1}\{y_i =k\}x_i}{\sum_{i=1}^{N}\mathbf{1}\{y_i =k\}}\\
\Sigma_k^{*} &=  \frac{\sum_{i=1}^{N}(x_i - \mu_k)(x_i - \mu_k)^\top\mathbf{1}\{y_i =k\}}{\sum_{i=1}^{N}\mathbf{1}\{y_i =k\}}
\end{align*}
$$

#### Inference

After we have obtained parameters, we can start to use our model to make predictions. For a given pixel sample $x$, we first compute the joint distribution against all classes, i.e. $p(x,y=1)$, $p(x,y=2)$, $p(x,y=3)$. The prediction $\hat{y} = \underset{\boldsymbol c}{\operatorname{argmax}} p(x,y=c)$.

### Bin Detection

Here we employee a two fold approach: first segment the image with only pixels of the same color as blue recycling bins, then draw bounding box on pixel regions that are mostly to be blue recycling bins.

#### Mask Segmentation

Based on the fact that blue recycling bin has a distinctive color, we can build a color classifier using approach outlined in part 1. The objective is simple: is the given pixel recycling bin blue or not. 

#### Training Data

First we label the recycling bin region in the image, this will be the training data for recycling bin class. The rest of the image will be training data for the none-recycle bin class. Additionally, sky blue are also sampled to create sky blue class to increase performance and avoid confusion between sky and recycling bin.

#### Color Space

Lighting conditions can affect RGB values dramatically. Therefore picking a different colorspace can make the classifier more robust and accurate in various conditions. Here we experiment with following colorspaces `["HSV","HLS","LAB","RGB","YUV"]`. We then measure the distance between $\mu_{\text{bin}}$ and $\mu_{\text{not_bin}}$, which indicates how far they are apart under that particular colorspace. When mean vectors are far apart, classifier is less likely to mislabel one class for another. 

|   Colorspace   | $\|\mu_{\text{bin}} - \mu_{\text{not_bin}}\|$  |
| :----: | :----: |
|   HSV   |  79.23    |
|   HLS   |   64.43   |
|   LAB   |   35.76   |
|   RGB   |   52.50   |
|   YUV   |   32.11   |

From this table, we can clearly see HSV is the best colorspace to use.

#### Image Denoising

The segmentation mask might not be perfect. Therefore denoising can help to remove pixels that are not part of the recycling bin. We choose to use `opening` from skimage, which removes small bright spots. As we can see below, before using `opening` there are 330 regions that can potentially be labeled as recyling bins. After `opening`, there are only 4 candidates left.

![before_opening](/Users/Charlie/Downloads/before_opening.png)

![before_opening](/Users/Charlie/Downloads/after_opening.png "title")

#### Bounding Box Rejection

Recycling bins has a particular size and shape. It's a tall rectangle that occupies significant parts of the image. Consequently, bounding boxes that are too small  or has the shape of a fat rectangle are unlikely to be recycling bins.  Here we propose the following algorithm:

1. Reject bounding box that covers less than $\frac{1}{200}$ of the entire image.
2. Reject bounding box whose width is larger than 1.5 times the height.
3. Among remaining bounding boxes, reject those that are less than $\frac{1}{10}$ the size of  the largest bounding box.

## Results
### Final Parameters - Color Classification 
```python
mean
red   : [0.75250609 0.34808562 0.34891229]
green : [0.35060917 0.73551489 0.32949353]
blue  : [0.34735903 0.33111351 0.73526495]

covariance
red  : [[0.0370867  0.01844078 0.01863285]
        [0.01844078 0.06201456 0.00858164]
        [0.01863285 0.00858164 0.06206846]]
green: [[0.05578115 0.01765327 0.00873955]
        [0.01765327 0.03481496 0.0170234 ]
        [0.00873955 0.0170234  0.05606864]]
blue : [[0.05458538 0.00855282 0.0171735 ]
        [0.00855282 0.05688308 0.01830849]
        [0.0171735  0.01830849 0.0357719 ]]
```

### Mask Segmentation
From left to right: image in RGB, image in HSV, segmentation mask, mask after filtering + bounding box detection

![Screen Shot 2021-01-26 at 5.20.32 PM](/Users/Charlie/Downloads/Screen Shot 2021-01-26 at 5.20.32 PM.png)

![Screen Shot 2021-01-26 at 10.32.52 PM](/Users/Charlie/Downloads/Screen Shot 2021-01-26 at 10.32.52 PM.png)

![Screen Shot 2021-01-26 at 5.20.42 PM](/Users/Charlie/Downloads/Screen Shot 2021-01-26 at 5.20.42 PM.png)

### Bounding Box Coordinates 
```text
validation coordinates:
0065.jpg : [[159, 319, 221, 472], [764, 418, 923, 620]]
0064.jpg : [[363, 114, 464, 263]]
0070.jpg : []
0066.jpg : []
0067.jpg : [[586, 307, 705, 503], [713, 307, 828, 508]]
0063.jpg : [[173, 95, 266, 230]]
0062.jpg : [[228, 184, 413, 437], [27, 362, 132, 496]]
0061.jpg : [[189, 152, 310, 294]]
0069.jpg : []
0068.jpg : []
```

### Conclusion

Gaussian classifier worked incredibly well on pixel classification, achieving 100% accuracy on the validation set. We think high accuracy is the result of ample training samples. There are about 1000 images in each class, which enables the classifier to learn the complete distribution of colors. 

Meanwhile, the same classifier design also performs well when segmenting the reclining bin regions. For most of the training and testing image, classifier is able to successfully segment relying bins in the image. The addition of sky blue class helps to reduce false positive samples when sky is in the background. However, this also cause recycling bins whose color are the same as sky to be rejected. 

After segmentation mask is produced, the proposed bounding box detection algorithm works well on the validation set, achieving 100% accuracies on all images. 





















