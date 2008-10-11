from numpy import *
import grid_gen
from PIL import Image
from ore.ore import *
from scipy import weave
from scipy.weave import converters


bld = SVOBuilder()


n = array([0, 0, 0, 0], int8)
#for i in xrange(16):
#    bld.BuildRange(4, point_3i(i, i, i), point_3i(1, 1, 1), array([0, 0, i*16, 255], uint8), n)
#    bld.BuildRange(4, point_3i(15-i, i, i), point_3i(1, 1, 1), array([0, i*16, i*16, 255], uint8), n)

bld.BuildRange(2, point_3i(3, 3, 3), point_3i(1, 1, 1), array([128, 128, 128, 255], uint8), n)

a = array([0, 0, 0, 0], uint8)
a = a.repeat(8)
a.shape = (2, 2, 2, 4)
print a
bld.BuildRange(2, point_3i(0, 0, 0), point_3i(2, 2, 2), a, n.repeat(8))


#bld.BuildRange(9, point_3i(0, 511, 0), point_3i(1, 1, 1), array([0, 255, 0, 255], uint8))
#bld.BuildRange(9, point_3i(0, 0, 511), point_3i(1, 1, 1), array([0, 0, 255, 255], uint8))

print bld.CalcLeafCount(), bld.nodecount

vt = VoxTree(bld)
vt.save("data/hmap.vox")
