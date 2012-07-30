import pyximport; pyximport.install()
import numpy as np
import pylab as pl
import t

dens = np.ones((200, 200), np.float32)
dens[10:20, 10:20] = -1
dens[100:150, 50:70] = 1.5
dens[30:-30, 150:170] = np.inf
dist, path = t.test1(dens)

pl.imshow(dist)
pl.contour(dist, 20, colors='black')
pl.show()



