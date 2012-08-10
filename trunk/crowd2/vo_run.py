import pyximport; pyximport.install()
import numpy as np
import pylab as pl
from matplotlib.animation import FuncAnimation

from vo_model import VOModelBase
import pathfield
from time import clock 


class VOModel(VOModelBase):
    def __init__(self):
        self.grid_step = 1.0
        self.time_step = 0.2

        obst_mask, exit_mask = pathfield.load_raster_map('data/vo.png')
        dist, path = pathfield.calc_path_map(obst_mask, exit_mask)
        self.pathmap = path
        self.obst_mask = obst_mask
        h, w = path.shape[:2]
        pos = np.random.rand(10000, 2)*(w-200, h-200) + (100, 100)
        x, y = np.int32(pos + 0.5).T
        m = obst_mask[y, x]
        pos = pos[~m]
        self.pos = np.float32(pos)

def plot_model(model):
    pl.imshow(model.obst_mask, cmap='gray_r', vmax=3.0, interpolation='nearest')
    x, y = np.asarray(model.pos).T
    pl.plot(x, y, '.')

if __name__ == '__main__':
    model = VOModel()
    fig = pl.figure()
    pl.imshow(model.obst_mask, cmap='gray_r', vmax=3.0, interpolation='nearest')

    x, y = model.pos.T
    agent_plt = pl.plot(x, y, '.')[0]
    def update(_):
        model.step()
        agent_plt.set_data(x, y)

    anim = FuncAnimation(fig, update, interval=50)

    pl.show()




        
