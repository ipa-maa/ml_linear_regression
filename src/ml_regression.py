
"""
Created on May 3, 2019

@author: maa
@attention: ml test for linear regression
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.0.0

#############################################################################################

History:
- v1.0.0: first init
"""

from misc import Font
from misc import printy
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import sympy as sp

from pointcloud import Pointcloud


class MLRegression:
    def __init__(self):
        self.pc = Pointcloud()
        self.params = {'m': 2.0,
                       'c': 2.0,
                       'size': 100,
                       'sigma': 17.0}
        self.y_rand = self._generate_points(**self.params)
        self.y = [self.params['m'] * x + self.params['c'] for x in range(len(self.y_rand))]

    def _generate_points(self, m: float, c: float, size: int, sigma: float) -> np.array:
        return self.pc.generate_points(m=m, c=c, size=size, sigma=sigma)

    def _calc_gradient(self):
        """
        TODO: use sympy lambdify to generate gradient function
        """
        pass

    def main(self):
        plt.plot([x for x in range(len(self.y_rand))], self.y_rand, 'bo')
        plt.bar([x for x in range(len(self.y_rand))], self.y_rand - self.y)
        plt.show()


if __name__ == "__main__":
    mlr = MLRegression()
    mlr.main()

    x, y, z = sp.symbols('x, y, z')
    sp.init_printing()
    eq1 = x**2 + y
    print(eq1.evalf(5, subs={x: 5.0, y: 2.0}))
