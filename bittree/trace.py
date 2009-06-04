from numpy import *
from pylab import *

from htree import *

def load_htree():
    bricks = fromfile("bit_bricks.dat", uint64)
    grids = fromfile("grids.dat", uint32)
    grids = grids.reshape(-1, GridSize ** 3)
    return (bricks, grids, len(grids)-1)

def make_rays(sx, sy, viewdir, fov):
    def norm_vec(v):
        return v / linalg.norm(v)
    fwd = norm_vec(viewdir)
    right = norm_vec( cross(viewdir, array([0.0, 0.0, 1.0])) )
    up = cross(right, fwd)
    dh = 2 * tan(radians(0.5*fov)) / sx

    vrayy, vrayx = ogrid[:sy, :sx]
    vrayx = (vrayx - 0.5*(sx-1)) * dh
    vrayy = (vrayy - 0.5*(sy-1)) * dh

    rays = [(vrayx * right[ax] + vrayy * up[ax] + fwd[ax])[..., newaxis] for ax in xrange(3)]
    return concatenate(rays, 2).astype(float32).copy()


def trace(htree, eye, target):
    eye = array(eye, float32)
    target = array(target, float32)
    rays = make_rays(512, 384, target - eye, 70)
    (bricks, grids, root) = htree
    hit = zeros((rays.shape[0], rays.shape[1]), int32)
    voxel_pos = zeros((rays.shape[0], rays.shape[1], 3), int32)

    from scipy import weave
    code = '''
      #line 38 "trace.py"
      RayTracer tracer( (const Brick *) bricks, (const Grid *) grids, root);
      point_3f veye(eye[0], eye[1], eye[2]);
      int rayNum = Nrays[0] * Nrays[1];
      for (int i = 0; i < rayNum; ++i)
      {
        point_3f dir(rays[3*i], rays[3*i+1], rays[3*i+2]);
        tracer.trace(veye, dir);
        hit[i] = tracer.hit ? 1 : 0;
        reinterpret_cast<point_3i*>(voxel_pos)[i] = tracer.voxelPos;
      }
    '''
    args = ["bricks", "grids", "root", "eye", "rays", "voxel_pos", "hit"]
    weave.inline(code, args, headers=['"stdafx.h"', '"bittree.h"'], include_dirs=['.', '../cpp'], force=1, compiler='msvc')
    return hit, voxel_pos


if __name__ == '__main__':
    print "loading htree"
    htree = load_htree()
    
#    eye = array([1.3, 1.5, 1.1])
#    target = array([0.5, 0.5, 0.5])
    eye = array([2.0, 3.0, 1.0])
    target = array([0.5, 0.5, 0.5])
    hit, voxel_pos = trace(htree, eye, target)
    imshow(hit)

    colorbar()
    show()

