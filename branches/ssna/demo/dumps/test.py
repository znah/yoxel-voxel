from numpy import *
from pylab import *
from scipy import signal

base_fn = "dmp_0_800x600"
size = (600, 800)
fov = 70.0

def calcDistCoefs(size, fov):
    dirs = calcViewSpaceRayDirs(size, fov)
    return sqrt((dirs*dirs).sum(-1))

def calcViewSpaceRayDirs(size, fov):
    ay, ax = mgrid[:size[0], :size[1]]
    ay -= size[0]/2
    ax -= size[1]/2
    step = tan( radians(fov/2) ) / (size[1]/2)
    ax = ax*step
    ay = ay*step
    res = empty( size+(3,), float32 )
    res[...,0], res[...,1], res[...,2] = ax, ay, 1.0
    return res


dist = fromfile(base_fn+".dist", float32)
dist.shape = size
dist[dist <= 0] = 10000;
dist /= calcDistCoefs(size, fov)

pos = fromfile(base_fn+".pos", float32)
pos.shape = size + (3,)

raydir = fromfile(base_fn+".dir", float32)
raydir.shape = size + (3,)

color = fromfile(base_fn+".color", uint8)
color.shape = size + (4,)

normal = fromfile(base_fn+".normal", float32)
normal.shape = size + (3,)

 
voxelSize = 0.5 ** 11
pixelAng = radians(fov/size[1])
scrVoxSize = voxelSize / (dist * pixelAng)

dotNV = maximum( 0.0, (-raydir*normal).sum(-1) )

viewRaydir = calcViewSpaceRayDirs(size, fov)


dstep = scrVoxSize.astype(int32)
pad = dstep.max()

y, x = mgrid[pad:size[0]-pad, pad:size[1]-pad]
dstep = dstep[y, x]
dstep = maximum(dstep, 1)

bufSize = dstep.shape

ndx = zeros(bufSize, float32)
ndy = zeros(bufSize, float32)
ndxa = zeros(bufSize, float32)
ndya = zeros(bufSize, float32)

for ang in arange(0, 2*pi, 2*pi / 10):
    dx, dy = cos(ang), sin(ang)
    
    idx = (dx*dstep).astype(int32)
    idy = (dy*dstep).astype(int32)

    dz = dist[y-idy, x-idx] - dist[y, x]
    dz = clip(dz, -3*voxelSize, 3*voxelSize)
    ndx += dx*dz
    ndy += dy*dz
    ndxa += abs(dx)
    ndya += abs(dy)
    print ang

ndx /= ndxa
ndy /= ndya
n = empty( bufSize + (3,), float32)
n[...,0] = -ndx
n[...,1] = -ndy
n[...,2] = 1

nl = sqrt((n*n).sum(-1))
nl = repeat(nl, 3)
nl.shape = n.shape
n /= nl

l = maximum( 0.0, (viewRaydir[y, x]*n).sum(-1) )

imshow(dotNV, origin="bottom", interpolation="nearest")
colorbar()
gray()
show()
