import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
import pylab

import htree


if __name__ == "__main__":
    print "loading data ..."
    alpha = np.fromfile("../data/bonsai.raw", uint8)
    alpha.shape = (256, 256, 256)

    hint_bricks = np.fromfile("hint_bricks.dat", uint64)
    hint_grids = np.fromfile("hint_grids.dat", uint32)
    hint_grids.shape = (-1, htree.GridSize**3)

    print "upload to gpu ..."
    d_alpha = cuda.to_device(alpha)
    d_hint_bricks = cuda.to_device(d_hint_bricks)
    d_hint_grids = cuda.to_device(d_hint_grids)

    mod = SourceModule("""




    __global__ void testKernel()




    """) 









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