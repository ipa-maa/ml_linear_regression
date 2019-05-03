"""
Created on May 3, 2019

@author: maa
@attention: pointcloud generation for ml regression
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.0.0

#############################################################################################

History:
- v1.0.0: first init
"""

import numpy as np


class Pointcloud:
    def __init__(self):
        pass

    def generate_points(self, m: float, c: float, size: int, sigma: float) -> np.array:
        """
        n (int): number of values to generate
        """
        x = np.arange(size)
        y = m * x + c
        mu = 0.0
        rand_y = y + np.random.normal(loc=mu, scale=sigma, size=len(y))
        return rand_y
