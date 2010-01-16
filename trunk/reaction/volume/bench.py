from numpy import *
import pycuda.driver as cuda
import pycuda.gpuarray as ga
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule

from time import clock


def divUp(a, b):
    return (a + b - 1) // b


mod = SourceModule('''
  typedef char data_t;

  texture<data_t, 1, cudaReadModeElementType> tex;

  __device__ int getTid()
  {
    int bsize = blockDim.x * blockDim.y;
    int bidx = blockIdx.x + blockIdx.y * gridDim.x;
    return threadIdx.x + bsize * bidx;
  }

  __global__ void ReadTest(int n, data_t * src, float * dst, int ofs)
  {
    int tid = getTid();
    int i = tid + ofs;
    if (tid >= n || i < 0 || i >= n)
      return;
    dst[tid] = src[i];
  }
  
  __global__ void ReadTestTex(int n, data_t * src, float * dst, int ofs)
  {
    int tid = getTid();
    int i = tid + ofs;
    if (tid >= n || i < 0 || i >= n)
      return;
    dst[tid] = tex1Dfetch(tex, i);
  }
''')
src_t = uint8

ReadTest = mod.get_function("ReadTest")
ReadTestTex = mod.get_function("ReadTestTex")
tex = mod.get_texref("tex")

print "reg: %d,  lmem: %d " % (ReadTest.num_regs, ReadTest.local_size_bytes)

n = 16 * 2**20
src = ga.to_gpu( arange(n, dtype = src_t) )
dst = ga.zeros((n,), float32)
src.bind_to_texref(tex)

def runKernel(bsize = 256, ofs = 0, func = ReadTest):
    bnum = divUp(n, bsize)
    func(int32(n), src, dst, int32(ofs), block = (bsize, 1, 1), grid = (1024, divUp(bnum, 1024)), texrefs = [tex])

runKernel(func = ReadTest)
print dst
runKernel(func = ReadTestTex)
print dst

def run(bsize = 256, ofs = 0, func = ReadTest):
    cuda.Context.synchronize()
    start = clock()
    for i in xrange(10):
        runKernel(bsize, ofs, func)
    cuda.Context.synchronize()
    stop = clock()

    return (stop - start) / 10 * 1000


blockSize = range(32, 512+1, 32)

def run2(ofs = 0, func = ReadTest):
    tt = []
    for bsize in blockSize:
        t = run(bsize, ofs, func)
        tt.append(t)
        print bsize, t
    return tt

import pylab

pylab.plot(blockSize, run2(0))
pylab.plot(blockSize, run2(1))
pylab.plot(blockSize, run2(0, ReadTestTex), '--')
pylab.plot(blockSize, run2(1, ReadTestTex), '--')
pylab.plot([0])
pylab.savefig("t.png")
pylab.show()








