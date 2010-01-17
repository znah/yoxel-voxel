from __future__ import with_statement
import sys
sys.path.append("..")
from zgl import *

import pycuda.driver as cu
import pycuda.gpuarray as ga
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule


class CuSparseVolume:
    def __init__(self, brickSize = 8):
        self.dtype = dtype(float32)
        self.brickSize = brickSize
        self.brickSize3 = brickSize**3
        self.brickMap = {}
        self.reallocPool(256)

    def reallocPool(self, capacity):
        d_bricks = {}
        d_bricks['data'] = ga.zeros((capacity,) + (self.brickSize,)*3, self.dtype)
        d_bricks['info'] = ga.zeros((capacity, 4), int32)
        if hasattr(self, 'd_bricks'):
            for name, d_src in self.d_bricks:
                d_dst = d_bricks[name]
                cu.memcpy_dtod(d_dst, d_src, min(d_dst.nbytes, d_src.nbytes))
                d_src.gpudata.free()
        self.d_bricks = d_bricks
        self.capacity = capacity

    def allocBrick(self, pos):
        pos = tuple(pos)
        if pos in self.brickMap:
            return self.brickMap[pos]

        brickN = len(self.brickMap)
        if brickN >= self.capacity:
            self.reallocPool(self.capacity * 2)

        self.brickMap[pos] = brickN
        return brickN

    def brickOfs(self, idx):
        return idx * self.brickSize3 * self.dtype.itemsize

    def __setitem__(self, pos, data):
        idx = self.allocBrick(pos)
        d_ptr = int(self.d_bricks['data'].gpudata) + self.brickOfs(idx)
        data = ascontiguousarray(data, self.dtype)
        cu.memcpy_htod(d_ptr, data)

    def __getitem__(self, pos):
        pos = tuple(pos)
        a = zeros((self.brickSize,)*3, self.dtype)
        if pos not in self.brickMap:
            return a
        idx = self.brickMap[pos]
        d_ptr = int(self.d_bricks['data'].gpudata) + self.brickOfs(idx)
        cu.memcpy_dtoh(a, d_ptr)
        return a


    commonCode = '''
        texture<float, 1> brick_data_tex;
        texture<int4, 1> brick_pos_tex;

        __constant__ float * brick_data_ptr;



    '''

    def makeProcessor(self, neib, proc):
        '''
          neib : 0, 1, 7 
        '''

        template = '''
          // block size 8x8x4

          const int brickSize = 8;
          const int brickSize3 = 8*8*8;

          typedef float value_t;

          texture<value_t, 1> brick_data_tex;
          texture<int4, 1>  brick_info_tex;

          __constant__ int brick_num;
          __constant__ float * brick_data_ptr;

          __device__ bool Proc(int4 brickInfo, int3 idx, value_t & output) 
          { 
            output = brickInfo.x + brickInfo.y + brickInfo.z + idx;
            return true; 
          }

          __global__ void RunKernel()
          {
            int brickId = blockIdx.x + blockIdx.y * gridDim.x;
            if (brickId > brick_num)
              return;

            int4 brickInfo = tex1Dfetch(brick_info_tex, brickId);
            value_t output;
            if (Proc(brickInfo, threadIdx, output))
              brick_data_ptr[brickId * brickSize3] = output;
          }

        '''

        



        
'''
class App(ZglAppWX):


    def __init__(self):
        ZglAppWX.__init__(self, viewControl = FlyCamera())
    
    def display(self):
        pass
'''
if __name__ == '__main__':
    #App().run()
    vol = CuSparseVolume()

    a = arange(8**3, dtype = float32)
    a.shape = (8, 8, 8)

    vol[1, 2, 3] = a
    b = vol[1, 2, 3]
    print abs(a - b).max()


