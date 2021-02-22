import numpy as np

class ParticleFilter:
    def __init__(self, n_particles=25, n_effective=0.2):
        self.n_particles = n_particles
        self.n_eff = int(self.n_particles * n_effective)
        self.particles = np.zeros([3, self.n_particles])
        self.alphas = np.ones(self.n_particles) * (1/self.n_particles)
        self.ones = np.ones_like(self.alphas)

    def predict_all(self, lienar_v, angular_v, t, sigma_linear=0.5, sigma_angular=0.5):
        # add noise
        lienar_v_noisy = lienar_v + np.random.normal(0, sigma_linear)
        angular_v_noisy = angular_v + np.random.normal(0, sigma_angular)

        theta = self.particles[-1,:]
        cos_all = lienar_v_noisy * np.cos(theta)
        sin_all = lienar_v_noisy * np.sin(theta)
        omega_all = self.ones * angular_v_noisy
        return self.particles + t * np.stack([cos_all,sin_all,omega_all])

    def predict(self, x, v, omega, t):
        """Use motion model to predict robot pose
        Note theta is in car frame.

        Args:
            x (np array): shape 1 x 3. Pose at current location. [x,y, theta (radians)]
            v (float): linear veloticy
            omega (float): angular velocity in radians
            t (float): time interval

        Returns:
            np array: shape 1x3, pose at new location [x,y, theta] 
        """
        theta = x[-1]
        return x + t * np.array([v*np.cos(theta), v*np.sin(theta), omega])

    def update(self):
        pass
    
    def resampling(self):
        pass

