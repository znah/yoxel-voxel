import pyximport; pyximport.install()
import numpy as np
import pylab as pl
import scipy.ndimage as nd
from PIL import Image

import crowd_cpp


def load_raster_map(fn):
    img = np.asarray( Image.open(fn) )[...,:3]
    h, w = img.shape[:2]
    obst_mask = (img == 0).all(-1)
    exit_mask = img[...,1] > img[...,0]
    return obst_mask, exit_mask

def calc_path_map(obst_mask, exit_mask):
    obst_dist = nd.distance_transform_edt(~obst_mask)
    obst_avoid_dist = 3.0
    dens = np.float32( 1.0 + 1.0 / ((np.minimum(obst_dist, obst_avoid_dist) + 1.0)*0.1) )
    dens[obst_mask] = np.inf
    dens[exit_mask] = -1
    dist, path = crowd_cpp._calc_distmap(dens)
    # TODO: navigation inside of obstacles
    return dist, path

if __name__ == '__main__':
    obst_mask, exit_mask = load_raster_map('data/vo.png')
    dist, path = calc_path_map(obst_mask, exit_mask)

    pl.imshow(dist, interpolation='nearest')
    pl.contour(dist, 100, colors='black')
    pl.show()



