from numpy import *
from ore.ore import *
import gc

bld = DynamicSVO()


start = (3, 4, 4)
end = (9, 6, 6)

bricks = [ (k, j, i) for k in range(start[0], end[0]) for j in range(start[1], end[1]) for i in range(start[2], end[2]) ]

#bricks = [(0, 0, 0), (0, 7, 7), (14, 7, 7)]


for k, j, i in bricks:
    fn = "data/VolumeData/d_0219_%04d" % (k*64 + j*8 + i)
    print "processing", k, i, j,
    try:
        data = fromfile(fn, uint8)
        data.shape = (128, 256, 256)
    except:
        print "error"
        continue

    data = data
    src = MakeIsoSource(point_3i(256, 256, 128), data)
    src.SetIsoLevel(200)
    #src.SetInside(True)

    bld.BuildRange(11, point_3i(i*256, j*256, k*128), BuildMode.GROW, src)
    print bld.nodecount

    del data
    del src

    gc.collect()



bld.Save("data/large_vol.vox")

