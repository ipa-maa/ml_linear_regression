
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
from ml_linear_regression.src.misc import Font
from ml_linear_regression.src.misc import printy
import numpy as np
from tensorflow import keras
import tensorflow as tf
from matplotlib import pyplot as plt
import sympy as sp


class Pointcloud:
    def __init__(self):
        self.seed = np.random.seed(42)
        self.params = {'m': 2.0,
                       'c': 2.0,
                       'size': 1000,
                       'sigma': 12.0}
        self.x = np.arange(0, self.params['size'])
        self.y = self.params['m'] * self.x + self.params['c']
        self.y_rand = self.y + np.random.normal(loc=0.0, scale=self.params['sigma'], size=len(self.y))


class MLRegression:
    def __init__(self):
        pc = Pointcloud()
        self.x = pc.x
        self.y = pc.y
        self.y_rand = pc.y_rand
        self.lr = 0.001
        self.network = self._build_network()

    def _build_network(self) -> keras.Model:
        input_shape = (1,)
        layer_input = keras.Input(shape=input_shape, name='layer_input')
        init1 = keras.initializers.glorot_normal(seed=13)
        init2 = keras.initializers.glorot_uniform(seed=13)
        layer_dense = keras.layers.Dense(units=100, activation='relu',
                                         kernel_initializer=init1, name='layer_dense')(layer_input)
        out = keras.layers.Dense(units=1, activation='linear', kernel_initializer=init2, name='out')(layer_dense)
        optimizer = keras.optimizers.RMSprop(lr=self.lr, rho=0.9)
        model = keras.Model(inputs=[layer_input], outputs=[out])
        model.compile(optimizer=optimizer, loss='mse')
        model.summary()
        return model

    def plot(self):
        plt.plot(self.x, self.y_rand, 'bo')
        plt.plot(self.x, self.y, 'r')
        plt.bar(self.x, self.y_rand - self.y)
        plt.show()


class MLRegression_tf:
    def __init__(self):
        pc = Pointcloud()
        self.x = pc.x
        self.y = pc.y
        self.y_rand = pc.y_rand
        self.lr = 0.001

        # with tf.device('/device:GPU:0'):
        self.X = tf.placeholder('float')
        self.Y = tf.placeholder('float')
        self.n = len(self.x)

        self.W = tf.Variable(np.random.rand(), name="W")
        self.b = tf.Variable(np.random.rand(), name="b")
        
        self.init = tf.global_variables_initializer()

    def _forward(self):
        return tf.add(tf.multiply(self.X, self.W), self.b)

    def _loss(self):
        # return tf.reduce_sum(tf.pow(self._forward() - self.Y, 2)) / self.n
        return tf.reduce_sum((self._forward() - self.Y) ** 2) / self.n

    def _create_optimizer(self):
        return tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self._loss())

    def main(self):
        with tf.Session() as sess:
            sess.run(self.init)
            optimizer = self._create_optimizer()

            for epoch in range(50):
                for _x, _y in zip(self.x, self.y_rand):
                    sess.run(optimizer, feed_dict={self.X: _x, self.Y: _y})

                if (epoch + 1) % 5 == 0:
                    c = sess.run(self._loss(), feed_dict={self.X: self.x, self.Y: self.y})
                    print("Epoch: {}, cost: {}, W: {}, b: {}".format(epoch, c, sess.run(self.W), sess.run(self.b)))


if __name__ == "__main__":
    if False:
        mlr = MLRegression()
        hist = mlr.network.fit(x=mlr.x, y=mlr.y_rand, batch_size=32, epochs=50, shuffle=True)
        plt.plot(mlr.x, mlr.y_rand, 'bo')
        y_predict = mlr.network.predict(mlr.x)
        plt.plot(mlr.x, y_predict, 'r')
        plt.show()
    else:
        mlr_tf = MLRegression_tf()
        mlr_tf.main()
