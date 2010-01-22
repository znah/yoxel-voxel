from __future__ import with_statement

from numpy import *
import pycuda.driver as cu
import pycuda.gpuarray as ga
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule

from cutypes import *
from string import Template
import os

import unittest
import pickle

class CuSparseVolume:

    def __init__(self, brickSize = 8, nhood = 6):
        self.dtype = dtype(float32)
        self.brickSize = brickSize
        assert nhood in [0, 6]
        self.nhood = nhood

        self.brickMap = {}
        self.nhoodDir = [
            ( 1, 0, 0 ),
            (-1, 0, 0 ),
            ( 0, 1, 0 ),
            ( 0,-1, 0 ),
            ( 0, 0, 1 ),
            ( 0, 0,-1 )]

        self.reallocPool(256)
        self.brickNum = 0

        self.Ctx = struct('Ctx',
            ( 'brick_num'  , c_int32         ),
            ( 'brick_data' , CU_PTR(c_float) ),
            ( 'brick_info' , CU_PTR(int4)    ),
            ( 'brick_nhood', CU_PTR(c_int32) ))

        brickSize2 = brickSize**2
        brickSize3 = brickSize**3
        Ctx_decl = gen_code(self.Ctx)
        self.header = '''
          typedef float value_t;
          const int bsize  = %(brickSize)d;
          const int bsize2 = %(brickSize2)d;
          const int bsize3 = %(brickSize3)d;
          const int nhood = %(nhood)d;

          texture<value_t, 1> brick_data_tex;
          texture<int4, 1>  brick_info_tex;
          texture<int, 1>   brick_nhood_tex;

          %(Ctx_decl)s
          __constant__ Ctx ctx;
        ''' % locals()

    def reallocPool(self, capacity):
        d_bricks = {}
        d_bricks['data'] = ga.zeros((capacity,) + (self.brickSize,)*3, self.dtype)
        d_bricks['info'] = ga.zeros((capacity, 4), int32)
        if self.nhood > 0:
            d_bricks['nhood'] = ga.zeros((capacity, self.nhood), int32)
        if hasattr(self, 'd_bricks'):
            for name in self.d_bricks:
                d_src = self.d_bricks[name]
                d_dst = d_bricks[name]
                cu.memcpy_dtod(d_dst.gpudata, d_src.gpudata, min(d_dst.nbytes, d_src.nbytes))
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

        if self.nhood > 0:
            d_nhoodPtr = int(self.d_bricks['nhood'].gpudata)
            
            nhoodIds = zeros(self.nhood, int32)
            apos = array(pos)
            for i, d in enumerate(self.nhoodDir):
                npos = tuple(apos + d)
                neibIdx = self.brickMap.get(npos, -1)
                nhoodIds[i] = neibIdx
                if neibIdx >= 0:
                    ofs = neibIdx * self.nhood + i^1
                    cu.memcpy_htod(d_nhoodPtr + ofs * sizeof(c_int32), c_int32(idx))
            ofs = idx * self.nhood
            cu.memcpy_htod(d_nhoodPtr + ofs * sizeof(c_int32), nhoodIds)

        self.brickMap[pos] = idx
        self.brickNum += 1
        return idx

    def brickOfs(self, idx):
        return idx * self.brickSize**3 * self.dtype.itemsize

    def __setitem__(self, pos, data):
        idx = self.allocBrick(pos)
        d_ptr = int(self.d_bricks['data'].gpudata) + self.brickOfs(idx)
        if isscalar(data):
            a = zeros((self.brickSize,)*3, self.dtype)
            a[:] = data
        else:
            a = ascontiguousarray(data, self.dtype)
        cu.memcpy_htod(d_ptr, a)

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
        ctx.brick_num = self.brickNum
        ctx.brick_data = self.d_bricks['data'].gpudata
        ctx.brick_info = self.d_bricks['info'].gpudata
        d_ctx = mod.get_global('ctx')[0]
        cu.memcpy_htod(d_ctx, ctx)

        brick_info_tex = mod.get_texref('brick_info_tex')
        self.d_bricks['info'].bind_to_texref_ext(brick_info_tex, channels = 4)
        
        brick_data_tex = mod.get_texref('brick_data_tex')
        self.d_bricks['data'].bind_to_texref_ext(brick_data_tex)

        brick_nhood_tex = mod.get_texref('brick_nhood_tex')
        self.d_bricks['nhood'].bind_to_texref_ext(brick_nhood_tex)

        func = mod.get_function(name)
        func(grid = (self.brickNum, 1), block=block)
        
    def processNeibAllocs(self):
        info = self.d_bricks['info'].get()[:self.brickNum]
        reqs = compress(info[:,3] != 0, info, 0)
        for i in xrange(self.nhood):
            rs = compress(info[:,3] & (1<<i), info, 0)
            pos = rs[:,:3] + self.nhoodDir[i]
            for p in pos:
                self[p] = 0
            
class Tests(unittest.TestCase):
    def __init__(self, *la, **ka):
        unittest.TestCase.__init__(self, *la, **ka)

        vol = CuSparseVolume()
        code = common_code + vol.header + '''
          #line 151
          extern "C" 
          __global__ void TestFill()
          {
            int bid = getbid();
            if (bid > ctx.brick_num)
              return;

            int4 info = tex1Dfetch(brick_info_tex, bid);
            uint3 cpos = threadIdx;
            value_t output = info.x + info.y + info.z + cpos.x + cpos.y + cpos.z;
            ctx.brick_data[bid * bsize3 + gettid()] = output;
          }

          __device__ uint calcOfs(int bid, int3 p)
          {
            return bid*bsize3 + p.z * bsize2 + p.y * bsize + p.x;
          }
          
          __device__ value_t fetchVol(int bid, int3 p)
          {
            return tex1Dfetch(brick_data_tex, calcOfs(bid, p));
          }

          __device__ value_t fetchNeib(int bid, int3 p)
          {
            if (p.x >= bsize)
              bid = tex1Dfetch(brick_nhood_tex, bid * nhood);
            if (p.x < 0)
              bid = tex1Dfetch(brick_nhood_tex, bid * nhood + 1);

            if (p.y >= bsize)
              bid = tex1Dfetch(brick_nhood_tex, bid * nhood + 2);
            if (p.y < 0)
              bid = tex1Dfetch(brick_nhood_tex, bid * nhood + 3);

            if (p.z >= bsize)
              bid = tex1Dfetch(brick_nhood_tex, bid * nhood + 4);
            if (p.z < 0)
              bid = tex1Dfetch(brick_nhood_tex, bid * nhood + 5);

            if (bid < 0)
                return 0.0f;

            p.x  = (p.x + bsize) % bsize;
            p.y  = (p.y + bsize) % bsize;
            p.z  = (p.z + bsize) % bsize;
            return fetchVol(bid, p);
          }

          extern "C" 
          __global__ void TestNeibMax()
          {
            int bid = getbid();
            if (bid > ctx.brick_num)
              return;

            int4 info = tex1Dfetch(brick_info_tex, bid);
            int3 bp = make_int3(info.x, info.y, info.z);
            int3 p = make_int3(threadIdx.x, threadIdx.y, threadIdx.z);
            
            float res = fetchVol(bid, p);
            res = max( res, fetchNeib(bid, p + make_int3( 1, 0, 0)) );
            res = max( res, fetchNeib(bid, p + make_int3(-1, 0, 0)) );
            res = max( res, fetchNeib(bid, p + make_int3( 0, 1, 0)) );
            res = max( res, fetchNeib(bid, p + make_int3( 0,-1, 0)) );
            res = max( res, fetchNeib(bid, p + make_int3( 0, 0, 1)) );
            res = max( res, fetchNeib(bid, p + make_int3( 0, 0,-1)) );
            ctx.brick_data[calcOfs(bid, p)] = res;
          }

          extern "C" 
          __global__ void TestMul()
          {
            int bid = getbid();
            if (bid > ctx.brick_num)
              return;

            int4 info = tex1Dfetch(brick_info_tex, bid);
            int3 bp = make_int3(info.x, info.y, info.z);
            int3 p = make_int3(threadIdx.x, threadIdx.y, threadIdx.z);
            
            float res = fetchVol(bid, p);
            ctx.brick_data[calcOfs(bid, p)] = res*2;
          }

          extern "C"
          __global__ void TestAllocReqest()
          {
            if (gettid() != 0)
              return;
            int bid = getbid();
            if (bid > ctx.brick_num)
              return;
            int flags = 0;
            for (int i = 0; i < nhood; ++i)
            {
              int neib = tex1Dfetch(brick_nhood_tex, bid * nhood + i);
              if (neib < 0)
                flags |= 1<<i;
            }
            ctx.brick_info[bid].w = flags;
          }


        '''
        self.mod = SourceModule(code, include_dirs = [os.getcwd()], no_extern_c = True)

    def testSetGet(self):
        vol = CuSparseVolume()
        a = arange(8**3, dtype = float32)
        a.shape = (8, 8, 8)
        vol[1, 2, 3] = a
        b = vol[1, 2, 3]
        self.assert_( all(a == b) )
        self.assert_( all(vol[5, 5, 3] == 0) )

    def testSetScalar(self):
        vol = CuSparseVolume()
        vol[3, -2, 1] = 5
        self.assert_( all(vol[3, -2, 1] == 5) )
        
    def testRunKernel(self):
        vol = CuSparseVolume()
        vol[0, 0, 0] = 0
        vol[1, 1, 1] = 0
        vol.runKernel(self.mod, "TestFill")
        a = sum( mgrid[:8,:8,:8], 0 )
        self.assert_( all( vol[0, 0, 0] == a ) )
        self.assert_( all( vol[1, 1, 1] == a+1+1+1 ) )

    def _testRealloc(self):
        vol = CuSparseVolume()
        bricks = pickle.load( file("bonsai02.dmp", "rb") )
        for p, a in bricks.items()[:600]:
           vol[p] = a
        for p, a in bricks.items()[:600]:
           self.assert_( all(vol[p] == a) )

    def testFetch(self):
        vol = CuSparseVolume()
        vol[1, 2, 3] = 1
        vol.runKernel(self.mod, "TestMul")
        self.assert_( all(vol[1, 2, 3] == 2) )

    def testNeib(self):
        vol = CuSparseVolume()
        p = array([5, 5, 5])
        vol[p] = 0
        mark = 1
        for dp in vol.nhoodDir:
            vol[p + dp] = mark
            mark += 1
        vol[p + (2, 0, 0)] = -1
        vol.runKernel(self.mod, "TestNeibMax")
        # TODO asserts
        #print vol[p + (2, 0, 0)]

    def testAllocReq(self):
        vol = CuSparseVolume()
        vol[0, 0, 0] = 0
        vol.runKernel(self.mod, "TestAllocReqest")
        vol.processNeibAllocs()
        self.assert_(vol.brickNum == 7)

        vol.runKernel(self.mod, "TestAllocReqest")
        vol.processNeibAllocs()
        self.assert_(vol.brickNum == 25)





if __name__ == '__main__':
    unittest.main()
