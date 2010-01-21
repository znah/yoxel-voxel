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
from string import Template

class CuSparseVolume:

    def __init__(self, brickSize = 8):
        self.dtype = dtype(float32)
        self.brickSize = brickSize
        self.brickMap = {}

        self.buildModule()
        self.reallocPool(256)

    def buildModule(self):
        self.CuCtx   = struct('CuCtx',
            ( 'brick_num' , c_int32         ),
            ( 'brick_data', CU_PTR(c_float) ),
            ( 'brick_info', CU_PTR(int4)    ))
        header = ''
        header += gen_code(range3i)
        header += gen_code(self.CuCtx)

        brickSize = self.brickSize
        brickSize2 = self.brickSize**2
        brickSize3 = self.brickSize**3
        
        code = '''
          %(header)s

          #line 43

          typedef float value_t;
          const int brickSize  = %(brickSize)d;
          const int brickSize2 = %(brickSize2)d;
          const int brickSize3 = %(brickSize3)d;

          texture<value_t, 1> brick_data_tex;
          texture<int4, 1>  brick_info_tex;

          __constant__ CuCtx ctx;

          
          __device__ bool inrange(range3i r, int3 p)
          {
            if (p.x < r.lo.x || p.y < r.lo.y || p.z < r.lo.z)
              return false;
            if (p.x >= r.hi.x || p.y >= r.hi.y || p.z >= r.hi.z)
              return false;
            return true;
          }
          
          __device__ bool Proc(int3 bpos, int flags, uint3 cpos, value_t & output) 
          { 
            output = bpos.x + bpos.y + bpos.z + cpos.x + cpos.y + cpos.z;
            return true; 
          }

          __global__ void RunKernel()
          {
            int brickId = blockIdx.x + blockIdx.y * gridDim.x;
            if (brickId > ctx.brick_num)
              return;

            int4 info = tex1Dfetch(brick_info_tex, brickId);
            int3 bpos = make_int3(info.x, info.y, info.z);
            value_t output;
            if (Proc(bpos, info.w, threadIdx, output))
            {
              int tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
              ctx.brick_data[brickId * brickSize3 + tid] = output;
            }
          }

        ''' % locals()
        print code
        self.mod = SourceModule(code)

    def runTest(self):
        ctx = self.CuCtx()
        n = len(self.brickMap)
        ctx.brick_num = n
        ctx.brick_data = self.d_bricks['data'].gpudata
        ctx.brick_info = self.d_bricks['info'].gpudata
        d_ctx = self.mod.get_global('ctx')[0]
        cu.memcpy_htod(d_ctx, ctx)

        brick_info_tex = self.mod.get_texref('brick_info_tex')
        self.d_bricks['info'].bind_to_texref_ext(brick_info_tex, channels = 4)

        RunKernel = self.mod.get_function("RunKernel")
        RunKernel(grid = (n, 1), block=(8, 8, 8))
        
        

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

        idx = len(self.brickMap)
        if idx >= self.capacity:
            self.reallocPool(self.capacity * 2)

        d_infoPtr = int(self.d_bricks['info'].gpudata) + idx * sizeof(int4)
        info = int4( pos[0], pos[1], pos[2], 0 )
        cu.memcpy_htod(d_infoPtr, info)

        self.brickMap[pos] = idx
        return idx

    def brickOfs(self, idx):
        return idx * self.brickSize**3 * self.dtype.itemsize

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
    vol[0, 0, 0] = zeros_like(a)
    b = vol[1, 2, 3]
    print abs(a - b).max()

    vol.runTest()
    print vol[0, 0, 0]
    print vol[1, 2, 3][0, 0, 0]



