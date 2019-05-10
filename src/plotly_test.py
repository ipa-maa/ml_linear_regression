import plotly.graph_objs as go
import plotly.io as pio
import plotly as py

import numpy as np


class Pointcloud:
    def __init__(self):
        self.seed = np.random.seed(42)
        self.params = {'m': 2.0,
                       'c': 2.0,
                       'size': 300,
                       'sigma': 12.0}
        self.x = np.arange(0, self.params['size'])
        self.y = self.params['m'] * self.x + self.params['c']
        self.y_rand = self.y + np.random.normal(loc=0.0, scale=self.params['sigma'], size=len(self.y))

    def plot(self):
        trace1 = go.Scatter(x=self.x, y=self.y, name='linear', mode='lines', line=dict(width=1))
        trace2 = go.Scatter(x=self.x, y=self.y_rand, name='noisy', mode='markers')
        layout = dict(title='plotly test',
                      xaxis=dict(title='x-axis'),
                      yaxis=dict(title='y-axis'),
                      margin=dict(l=50, r=80, t=50, b=50))
        fig = dict(data=[trace2, trace1], layout=layout)
        py.offline.plot(fig, 'test.html')
        pio.write_image(fig, 'test.pdf', scale=1)


if __name__ == "__main__":
    pc = Pointcloud()
    pc.plot()
