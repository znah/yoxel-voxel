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
import os

class CuSparseVolume:

    def __init__(self, brickSize = 8):
        self.dtype = dtype(float32)
        self.brickSize = brickSize
        self.brickMap = {}

        self.reallocPool(256)

        self.Ctx   = struct('Ctx',
            ( 'brick_num' , c_int32         ),
            ( 'brick_data', CU_PTR(c_float) ),
            ( 'brick_info', CU_PTR(int4)    ))

        brickSize2 = brickSize**2
        brickSize3 = brickSize**3
        Ctx_decl = gen_code(self.Ctx)
        self.header = '''
          typedef float value_t;
          const int bsize  = %(brickSize)d;
          const int bsize2 = %(brickSize2)d;
          const int bsize3 = %(brickSize3)d;

          texture<value_t, 1> brick_data_tex;
          texture<int4, 1>  brick_info_tex;

          %(Ctx_decl)s
          __constant__ Ctx ctx;
        ''' % locals()

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

    def runKernel(self, mod, name, block = None):
        if block is None:
            block = (self.brickSize,)*3
        
        ctx = self.Ctx()
        n = len(self.brickMap)
        ctx.brick_num = n
        ctx.brick_data = self.d_bricks['data'].gpudata
        ctx.brick_info = self.d_bricks['info'].gpudata
        d_ctx = mod.get_global('ctx')[0]
        cu.memcpy_htod(d_ctx, ctx)

        brick_info_tex = mod.get_texref('brick_info_tex')
        self.d_bricks['info'].bind_to_texref_ext(brick_info_tex, channels = 4)

        func = mod.get_function(name)
        func(grid = (n, 1), block=block)
        
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
    print abs(a - b).max() == 0


    code = common_code + vol.header + '''
      __device__ bool Proc(int3 bpos, int flags, uint3 cpos, value_t & output) 
      { 
        output = bpos.x + bpos.y + bpos.z + cpos.x + cpos.y + cpos.z;
        return true; 
      }

      extern "C" 
      __global__ void Test()
      {
        int bid = blockIdx.x + blockIdx.y * gridDim.x;
        if (bid > ctx.brick_num)
          return;
        int tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

        int4 info = tex1Dfetch(brick_info_tex, bid);
        int3 bpos = make_int3(info.x, info.y, info.z);
        value_t output;
        if (Proc(bpos, info.w, threadIdx, output))
          ctx.brick_data[bid * bsize3 + tid] = output;
      }
    '''
    #print code
    mod = SourceModule(code, include_dirs = [os.getcwd()], no_extern_c = True)
    vol.runKernel(mod, "Test")

    print vol[0, 0, 0][7, 7, 7] == 21
    print vol[1, 2, 3][0, 0, 0] == 1+2+3



