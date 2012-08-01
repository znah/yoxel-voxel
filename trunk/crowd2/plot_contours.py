import numpy as np
import pylab as pl
from matplotlib.collections import PolyCollection

from pathfield import load_raster_map

imgfn = 'data/vo.png'
cntfn = 'data/vo_contours.txt'
#imgfn = 'data/vo2.png'
#cntfn = 'data/vo2_contours.txt'

obst_mask, exit_mask = load_raster_map(imgfn)
contours = []
for s in open(cntfn):
    cnt = np.fromstring(s, sep=' ').reshape(-1, 2)
    contours.append(cnt)

pl.imshow(obst_mask, cmap='gray_r', vmax=3.0, interpolation='nearest')
polys = PolyCollection(contours, facecolors='none')
pl.gca().add_collection(polys)
pl.show()

    




