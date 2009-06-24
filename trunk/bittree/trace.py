import os

import pycuda.driver as cuda
import pycuda.gpuarray as ga
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
import pylab

import htree


if __name__ == "__main__":
    print "loading data ..."
    alpha = np.fromfile("../data/bonsai.raw", np.uint8)
    alpha.shape = (256, 256, 256)

    hint_bricks = np.fromfile("hint_bricks.dat", np.uint64)
    hint_grids = np.fromfile("hint_grids.dat", np.uint32)
    hint_grids.shape = (-1, htree.GridSize**3)

    print "uploading to gpu ..."
    d_alpha = cuda.to_device(alpha)
    d_hint_bricks = cuda.to_device(hint_bricks)
    d_hint_grids = cuda.to_device(hint_grids)

    src = file("trace.cu").read()
    mod = SourceModule(src, no_extern_c = True, include_dirs = [os.getcwd()]) 
    func = mod.get_function("TestFetch")
    print "TestFetch reg num:", func.num_regs

    '''
    struct RenderParams
    {
      node_id hintTreeRoot;
      uint2 viewSize;
    };
    '''
    render_params_t = np.dtype( [ ("hintTreeRoot", np.uint32), ("viewSize", np.uint32, 2) ] )
    render_params = np.zeros((1,), render_params_t)[0] # how to do it easier

    render_params["hintTreeRoot"] = hint_grids.shape[0]-1
    render_params["viewSize"][0] = 256
    render_params["viewSize"][1] = 256
    cuda.memcpy_htod(mod.get_global("rp")[0], render_params)

    d_dst = ga.empty((256, 256), np.uint32)
    func(np.float32(0.3), d_dst, block = (8, 8, 1), grid=(32, 32, 1))









'''
a = numpy.random.randn(4,4)

a = a.astype(numpy.float32)

a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)

cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
    __global__ void doublify(float *a)
    {
      int idx = threadIdx.x + threadIdx.y*4;
      a[idx] *= 2;
    }
    """)

func = mod.get_function("doublify")
func(a_gpu, block=(4,4,1))

a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print "original array:"
print a
print "doubled with kernel:"
print a_doubled
'''