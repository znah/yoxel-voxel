from __future__ import with_statement
import sys
sys.path.append("..")
from zgl import *

import pycuda.driver as cu
import pycuda.gpuarray as ga
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule

from cutypes import *

class CuSparseVolume:
    def __init__(self, brickSize = 8):
        self.dtype = dtype(float32)
        self.brickSize = brickSize
        self.brickSize3 = brickSize**3
        self.brickMap = {}
        self.reallocPool(256)

        range3i = struct('range3i', ('lo', int3), ('hi', int3))
        CuCtx   = struct('CuCtx',
            ( 'brick_num' , c_int32         ),
            ( 'brick_data', CU_PTR(c_float) ),
            ( 'brick_info', CU_PTR(int4)    ),
            ( 'range', range3i))

        self.cuCtx = CuCtx;
        self.header = ''
        self.header += gen_code(range3i)
        self.header += gen_code(CuCtx)
        self.header += '''
          typedef float value_t;
          const int brickSize = %(brickSize)d;
          const int brickSize3 = %(brickSize3)d;

          texture<value_t, 1> brick_data_tex;
          texture<int4, 1>  brick_info_tex;

          __constant__ CuCtx ctx;

        ''' % {"brickSize" : brickSize, "brickSize3" : brickSize**3}

        print self.header



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

    def makeProcessor(self, proc):

        code = self.header + '''
          __device__ bool inrange(range3i r, int3 p)
          {
            if (p.x < r.lo.x || p.y < r.lo.y || p.z < r.lo.z)
              return false;
            if (p.x >= r.hi.x || p.y >= r.hi.y || p.z >= r.hi.z)
              return false;
            return true;
          }
          
          __device__ bool Proc(int3 brickPos, int flags, int3 idx, value_t & output) 
          { 
            output = brickInfo.x + brickInfo.y + brickInfo.z + idx;
            return true; 
          }

          __ghibal__ void RunKernel()
          {
            int brickId = blockIdx.x + blockIdx.y * gridDim.x;
            if (brickId > brick_num)
              return;

            int4 info = tex1Dfetch(brick_info_tex, brickId);
            int3 pos = make_int3(info.x, info.y, info.z);
            if (!inrange(ctx.range, pos))
              return;
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


