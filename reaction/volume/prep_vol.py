from __future__ import division
from numpy import *
import pickle

a = fromfile("../img/bonsai.raw", uint8).reshape( (256,)*3 )
a = swapaxes(a, 0, 1)

brickSize = 8
gridSize = 256 / brickSize

gridShape = (gridSize,)*3

bricks = {}
for p in ndindex(*gridShape):
    bz, by, bx = array(p) * brickSize
    brick = a[bz:, by:, bx:] [:brickSize, :brickSize, :brickSize].astype(float32)
    brick /= 255.0
    brick = maximum( (brick-0.2)/0.8, 0)
    if brick.max() > 0:
      z, y, x = p
      bricks[(x, y, z)] = brick.copy()
      if len(bricks) % 1000 == 0:
          print len(bricks)

pickle.dump(bricks, file('bonsai02.dmp', 'wb'), -1)

print len(bricks) / prod(gridShape)
print len(bricks) * brickSize**3 * 4 / 2**20, 'mb'

 