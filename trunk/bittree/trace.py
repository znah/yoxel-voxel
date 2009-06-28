import os

import pycuda.driver as cuda
import pycuda.gpuarray as ga
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
import pylab

import htree


def makeViewToWldMtx(eye, target, up):
    def normalize(x):
        return x / np.linalg.norm(x)
    vdir = normalize(target - eye)
    vright = normalize(np.cross(vdir, up))
    vup = np.cross(vright, vdir)
    m = np.matrix(np.zeros((4, 4)), np.float32)
    m[:3,0].flat = vright
    m[:3,1].flat = vup
    m[:3,2].flat = -vdir
    m[:3,3].flat = eye
    m[3, 3] = 1
    return m;

if __name__ == "__main__":
    print "loading data ..."
    #alpha = np.fromfile("../data/bonsai.raw", np.uint8)
    #alpha.shape = (256, 256, 256)

    hint_bricks = np.fromfile("hint_bricks.dat", np.uint64)
    hint_grids = np.fromfile("hint_grids.dat", np.uint32)
    hint_grids.shape = (-1, htree.GridSize**3)

    print "uploading to gpu ..."
    #d_alpha = cuda.to_device(alpha)
    d_hint_bricks = cuda.to_device(hint_bricks)
    d_hint_grids = cuda.to_device(hint_grids)

    print "compiling kernel ..."
    src = file("trace.cu").read()
    mod = SourceModule(src, no_extern_c = True, include_dirs = [os.getcwd()]) 
    TestFetch = mod.get_function("TestFetch")
    print "TestFetch reg num:", TestFetch.num_regs
    Trace = mod.get_function("Trace")
    print "Trace reg num:", Trace.num_regs

    '''
    struct RenderParams
    {
      node_id hintTreeRoot;
      uint2 viewSize;

      float fovCoef; // tan(fov/2)

      float3 eyePos;
      float3x4 viewToWldMtx;
      float3x4 wldToViewMtx;

    };
    '''
    float3_t = np.dtype( (np.float32, 3) )
    float3x4_t = np.dtype( (np.float32, (3, 4)) )

    render_params_t = np.dtype([ 
      ("hintTreeRoot", np.uint32), 
      ("viewSize", np.uint32, 2), 
      ("fovCoef", np.float32), 
      ("eyePos", float3_t), 
      ("viewToWldMtx", float3x4_t), 
      ("wldToViewMtx", float3x4_t), 
    ])
    render_params = np.zeros((1,), render_params_t)[0] # how to do it easier

    render_params["hintTreeRoot"] = hint_grids.shape[0]-1
    render_params["viewSize"][:] = (512, 512)
    render_params["fovCoef"] = np.tan(np.radians( 45.0 / 2 ))

    eyePos = np.array([2.5, 0.8, 0.9])
    targetPos = np.array([0.5, 0.5, 0.5])
    v2wMtx = makeViewToWldMtx(eyePos, targetPos, np.array([0, 0, 1]))
    w2vMtx = np.linalg.inv(v2wMtx)
    render_params["eyePos"][:] = eyePos
    render_params["viewToWldMtx"][:] = v2wMtx[:3]
    render_params["wldToViewMtx"][:] = w2vMtx[:3]

    cuda.memcpy_htod(mod.get_global("rp")[0], render_params)

    hint_grid_tex = mod.get_texref("hint_grid_tex")
    hint_brick_tex = mod.get_texref("hint_brick_tex")
    hint_grid_tex.set_address(d_hint_grids, len(hint_grids.data))
    hint_brick_tex.set_address(d_hint_bricks, len(hint_bricks.data))
    hint_brick_tex.set_format(cuda.array_format.UNSIGNED_INT32, 2)

    dst = np.zeros((512, 512), np.float32)
    print "running kernel"
    #TestFetch(np.float32(0.6), cuda.Out(dst), block = (8, 8, 1), grid=(32, 32), texrefs = [hint_grid_tex, hint_brick_tex])
    Trace(cuda.Out(dst), block = (16, 16, 1), grid=(32, 32), texrefs = [hint_grid_tex, hint_brick_tex])

    def vis():
        import pylab
        pylab.imshow(dst, origin="bottom")
        pylab.colorbar()
