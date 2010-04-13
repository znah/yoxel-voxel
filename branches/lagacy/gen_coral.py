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

