
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

#####################################
#####################################
import sys
sys.path.append(".")
#####################################
#####################################
# %%
from ml_linear_regression.src.pointcloud import Pointcloud
import sympy as sp
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from ml_linear_regression.src.misc import printy
from ml_linear_regression.src.misc import Font


class MLRegression:
    def __init__(self):
        self.pc = Pointcloud()
        self.params = {'m': 2.0,
                       'c': 2.0,
                       'size': 100,
                       'sigma': 17.0}
        self.y_rand = self._generate_points(**self.params)
        self.y = [self.params['m'] * x + self.params['c'] for x in range(len(self.y_rand))]
        self.network = self._build_network()
        self.seed = np.random.seed(42)

    def _generate_points(self, m: float, c: float, size: int, sigma: float) -> np.array:
        return self.pc.generate_points(m=m, c=c, size=size, sigma=sigma)

    def _build_network(self) -> keras.Model:
        input_shape = (2,)
        layer_input = keras.Input(shape=input_shape, name='layer_input')
        layer_flatten = keras.layers.Flatten()(layer_input)
        layer_dense = keras.layers.Dense(units=100, activation='relu', name='layer_dense')(layer_flatten)
        out_m = keras.layers.Dense(units=1, activation='linear', name='out_m')(layer_dense)
        out_c = keras.layers.Dense(units=1, activation='linear', name='out_c')(layer_dense)
        optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)
        model = keras.Model(inputs=[layer_input], outputs=[out_m, out_c])
        model.compile(optimizer=optimizer, loss=self._loss(m=self.params['m'], c=self.params['c']))
        model.summary()
        return model

    def _loss(self, m: float, c: float):
        """
        y_true (tf tensor): target tensor
        y_pred (tf tensor): model output tensor
        """
        K = keras.backend

        def keras_loss(y_true, y_pred):
            # y_true = keras.backend.placeholder(shape=(1,1))
            y = [m, c]
            return K.mean(K.square(y_true - y_pred))
        return keras_loss

    def _calc_gradient(self):
        """
        TODO: use sympy lambdify to generate gradient function
        """
        pass

    def main(self):
        plt.plot([x for x in range(len(self.y_rand))], self.y_rand, 'bo')
        plt.bar([x for x in range(len(self.y_rand))], self.y_rand - self.y)
        plt.show()


# %%
mlr = MLRegression()
mlr.main()


# %%
