from numpy import *
from ore.ore import *
import gc


bld = DynamicSVO()

for z, y, x in ndindex(2, 2, 2):
    fn = "a_%d%d%d.npy" % (z, y, x)
    data = load(fn)
    src = MakeIsoSource(point_3i(512, 512, 512), data)
    src.SetIsoLevel(64)
    #src.SetInside(True)
    bld.BuildRange(11, point_3i(512*x, 512*y, 512*z), BuildMode.GROW, src)
    print fn

bld.Save("data/coral.vox")


'''
bld2 = DynamicSVO()

data = load('a_256.npy')
src = MakeIsoSource(point_3i(256, 256, 256), data)
src.SetIsoLevel(64)
bld2.BuildRange(11, point_3i(0, 0, 0), BuildMode.GROW, src)

bld2.Save("data/coral2.vox")
'''