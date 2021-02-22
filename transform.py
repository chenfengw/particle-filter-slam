import numpy as np

class Transform():
    def __init__(self):
        R_lidar = np.array([[ 1.30201e-03,  7.96097e-01,  6.05167e-01],
                            [ 9.99999e-01, -4.19027e-04, -1.60026e-03],
                            [-1.02038e-03,  6.05169e-01, -7.96097e-01]])
        p_lidar = np.array([0.8349, -0.0126869, 1.76416])
        self.car_T_lidar = self.calcualte_pose(R_lidar,p_lidar)

    def lidar_to_world(self,lidar_data,particle):
        """transform lidar data from lidar frame to world frame

        Args:
            lidar_data (np array): 4 x n_samples
            particle (np array): row vector. len 3, [x,y,theta]

        Returns:
            np array: lidar coordinates in world frame. 4 x n_samplese
        """
        # get car position
        xy = particle[:2]
        p = np.ones(3)
        p[:2] = xy

        # get car orientation
        theta = particle[-1]
        R = self.get_rotation_matrix(theta)
        world_T_car = self.calcualte_pose(R,p)

        return world_T_car @ self.car_T_lidar @ lidar_data

    @staticmethod
    def calcualte_pose(R,p):
        # assert R.shape == (3,3), "R must be 3x3"
        # assert len(p) == 3, "p must be row vector of len 3"
        T = np.zeros([4,4])
        T[:3,:3] = R
        T[:3,-1] = p
        T[-1,-1] = 1
        return T

    @staticmethod
    def get_rotation_matrix(theta):
        """Calculate rotation matrix around z axis rotated by angle theta

        Args:
            theta (float): rotation angle in radians

        Returns:
            np array: 3x3 rotation matrix
        """
        R = np.zeros([3,3])
        R_2d = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
        R[:2,:2] = R_2d
        R[-1,-1] = 1
        return R