import numpy as np

class ParticleFilter:
    def __init__(self):
        pass

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


class Particle:
    def __init__(self,location):
        pass
    
    def get_xy(self):
        pass

    def get_orientation(self):
        pass
