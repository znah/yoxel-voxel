import pyximport; pyximport.install()
import numpy as np
import pylab as pl
import scipy.ndimage as nd

import crowd_cpp

from PIL import Image

img = np.asarray( Image.open('data/vo.png') )[...,:3]
h, w = img.shape[:2]
obst_mask = (img == 0).all(-1)
exit_mask = img[...,1] > img[...,0]

obst_dist = nd.distance_transform_edt(~obst_mask)

obst_avoid_dist = 3.0
dens = np.float32( 1.0 + 1.0 / ((np.minimum(obst_dist, obst_avoid_dist) + 1.0)*0.1) )
dens[obst_mask] = np.inf
dens[exit_mask] = -1
dist, path = crowd_cpp._calc_distmap(dens)

pl.figure()
pl.imshow(dens, interpolation='nearest')
pl.colorbar()
pl.figure()
pl.imshow(dist, interpolation='nearest')
pl.contour(dist, 100, colors='black')
pl.show()



