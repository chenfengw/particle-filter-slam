import numpy as np

class ParticleFilter:
    def __init__(self, n_particles=25, threshold=0.2, add_noise=False):
        self.n_particles = n_particles
        self.n_thresh = int(self.n_particles * threshold)
        self.particles = np.zeros([3, self.n_particles])
        self.alphas = np.ones(self.n_particles) * (1/self.n_particles)
        self.ones = np.ones_like(self.alphas)
        self.add_noise = add_noise

    def predict_all(self, lienar_v, angular_v, t, sigma_linear=0.5, sigma_angular=0.5):
        # add noise
        if self.add_noise:
            lienar_v_noisy = lienar_v + np.random.normal(0, sigma_linear)
            angular_v_noisy = angular_v + np.random.normal(0, sigma_angular)
        else:
            lienar_v_noisy = lienar_v
            angular_v_noisy = angular_v

        # use motion model
        theta = self.particles[-1,:]
        return self.particles + t * np.vstack([lienar_v_noisy * np.cos(theta),
                                              lienar_v_noisy * np.sin(theta),
                                              self.ones * angular_v_noisy])

    def predict(self, x, v, omega, t):
        """Use motion model to predict robot pose
        Note theta is in car frame.

        Args:
            x (np array): shape 1 x 3. Pose at current location. [x,y, theta (radians)]
            v (float): linear velocity
            omega (float): angular velocity in radians
            t (float): time interval

        Returns:
            np array: shape 1x3, pose at new location [x,y, theta] 
        """
        theta = x[-1]
        return x + t * np.array([v*np.cos(theta), v*np.sin(theta), omega])

    def update(self, lidar_scan, myMap, tf):
        # lidar scan is in sensor frame

        # new alpha
        alpha_new = np.zeros(self.n_particles)
        correlations = []

        for idx in range(self.n_particles):
            the_particle = self.particles[:,idx]
            lidar_world = tf.lidar_to_world(lidar_scan, the_particle)

            # calculate map correlation
            c = myMap.map_correlation(lidar_world)
            correlations.append(c)

            # update alpha
            alpha_new[idx] = c.max()
        
        self.alphas = self.soft_max(alpha_new)
        max_idx = self.alphas.argmax()
        return max_idx, correlations[max_idx]

    def resampling(self):
        n_eff = 1 / np.linalg.norm(self.alphas)**2
        if n_eff < self.n_thresh:
            sample_idx = np.random.choice(self.n_particles,
                                          size=self.n_particles,
                                          replace=True,
                                          p=self.alphas)
            self.particles = self.particles[:,sample_idx]
            self.alphas = self.ones * (1 / self.n_particles)
            
    @staticmethod
    def soft_max(x):
        temp = np.exp(x)
        return temp / temp.sum()