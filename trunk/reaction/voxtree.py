from __future__ import with_statement
import os
from zgl import *
from cutools import *
import pycuda.gl as cuda_gl
from voxelizer import *


code = '''
#line 11
typedef unsigned int uint;

const int BRICK_SIZE   = 8;
const int VOL_SIZE     = 1024;
const int GRID_SIZE    = VOL_SIZE / BRICK_SIZE;
const int SLICE_STRIDE = VOL_SIZE * VOL_SIZE;
const int SLICE_NUM    = VOL_SIZE / 128;
const int MERGE_SIZE   = (BRICK_SIZE+1) * (BRICK_SIZE+1); // = 81
const int PAD_MERGE_SIZE = 128; 
const int COLUMN_WORDS = VOL_SIZE / 32;  // 32 max

__shared__ uint  s_column0 [ COLUMN_WORDS  ];
__shared__ uint  s_column1 [ COLUMN_WORDS  ];
__shared__ uint4 s_data0 [ MERGE_SIZE ];
__shared__ uint4 s_data1 [ MERGE_SIZE ];

extern "C"
__global__ void MarkBricks(const uint4 *g_slices, uint * g_mixedFlags, uint * g_solidFlags, uint * debug)
{
  int tid = threadIdx.x + threadIdx.y * (BRICK_SIZE+1);
  if (tid >= MERGE_SIZE)
    return;

  int x = threadIdx.x + blockIdx.x * BRICK_SIZE;
  int y = threadIdx.y + blockIdx.y * BRICK_SIZE;
  int srcOfs = x + y * VOL_SIZE;
  bool valid = x < VOL_SIZE && y < VOL_SIZE;
  
  // merge XY
  for (int i = 0; i < SLICE_NUM; ++i, srcOfs += SLICE_STRIDE)
  {
    s_data0[tid] = valid ? g_slices[srcOfs] : make_uint4(0, 0, 0, 0);
    s_data1[tid] = s_data0[tid];

    __syncthreads();

    for(uint s = PAD_MERGE_SIZE/2; s>0; s>>=1) 
    {
      int ts = tid + s;
      if (tid < s && ts < MERGE_SIZE) 
      {
        s_data0[tid].x |= s_data0[ts].x;
        s_data0[tid].y |= s_data0[ts].y;
        s_data0[tid].z |= s_data0[ts].z;
        s_data0[tid].w |= s_data0[ts].w;

        s_data1[tid].x &= s_data1[ts].x;
        s_data1[tid].y &= s_data1[ts].y;
        s_data1[tid].z &= s_data1[ts].z;
        s_data1[tid].w &= s_data1[ts].w;
      }
      __syncthreads();
    }

    if (tid == 0)
    {
      s_column0[i*4  ] = s_data0[0].x;
      s_column0[i*4+1] = s_data0[0].y;
      s_column0[i*4+2] = s_data0[0].z;
      s_column0[i*4+3] = s_data0[0].w;
      s_column1[i*4  ] = s_data1[0].x;
      s_column1[i*4+1] = s_data1[0].y;
      s_column1[i*4+2] = s_data1[0].z;
      s_column1[i*4+3] = s_data1[0].w;
    }
  }

  //if (tid < COLUMN_WORDS)
  //{
  //  uint ofs = (blockIdx.x + blockIdx.y * GRID_SIZE) * COLUMN_WORDS + tid*4;
  //  debug[ofs]
  //}


  // merge Z
  if (tid < COLUMN_WORDS) // 32 max
  {
    uint bits, next, packed;
    
    bits = s_column0[tid];
    next = 0;
    if (tid < COLUMN_WORDS-1)
      next = s_column0[tid+1];

    // propagate overlapping bits
    bits |= next << 31;
    bits |= bits >> 1;

    bits |= bits >> 1;
    bits |= bits >> 2;
    bits |= bits >> 4;
    packed = (bits & 1) + ((bits>>7) & 2) + ((bits>>14) & 4) + ((bits>>21) & 8);
    s_column0[tid] = packed;


    bits = s_column1[tid];
    next = 0;
    if (tid < COLUMN_WORDS-1)
      next = s_column1[tid+1];

    // propagate overlapping bits
    bits &= next << 31;
    bits &= bits >> 1;

    bits &= bits >> 1;
    bits &= bits >> 2;
    bits &= bits >> 4;
    packed = (bits & 1) + ((bits>>7) & 2) + ((bits>>14) & 4) + ((bits>>21) & 8);
    s_column1[tid] = packed;
  }

  if (tid < 4)
  {
    uint res0 = 0, res1 = 0;
    for (int i = 0; i < 8; ++i)
    {
      res0 |= s_column0[tid*8 + i] << (i*4);
      res1 |= s_column1[tid*8 + i] << (i*4);
    }
    int ofs = (blockIdx.x + blockIdx.y * GRID_SIZE) * 4 + tid;
    g_mixedFlags[ofs] = res0 ^ res1;
    g_solidFlags[ofs] = res1;
  }
}


'''
    
class App(ZglAppWX):
    z = Range(0.0, 1.0, 0.5)

    def __init__(self):
        ZglAppWX.__init__(self, viewControl = OrthoCamera())
        import pycuda.gl.autoinit

        size = 1024
        block_size = 8
        grid_size = size / block_size
        voxelizer = Voxelizer(size)
        v, f = load_obj("t.obj")
        with voxelizer:
            drawArrays(GL_TRIANGLES, verts = v, indices = f)
        
        gl2cudaBuf = cuda_gl.BufferObject(voxelizer.dumpToPBO().handle)

        mod = SourceModule(code, include_dirs = [os.getcwd(), os.getcwd()+'/include'], no_extern_c = True)
        MarkBricks = mod.get_function('MarkBricks') 
        print MarkBricks.num_regs, MarkBricks.shared_size_bytes

        mixedBits = ga.zeros((grid_size, grid_size, grid_size/32), uint32)
        solidBits = ga.zeros((grid_size, grid_size, grid_size/32), uint32)

        debug = ga.zeros((grid_size, grid_size, size/32), uint32)

        gl2cudaMap = gl2cudaBuf.map()
        MarkBricks(int32(gl2cudaMap.device_ptr()), mixedBits, solidBits, debug, block = (9, 9, 1), grid = (grid_size, grid_size))

        t = clock()
        for i in xrange(100):
            MarkBricks(int32(gl2cudaMap.device_ptr()), mixedBits, solidBits, debug, block = (9, 9, 1), grid = (grid_size, grid_size))
        print (clock() - t) / 100.0 * 1000
        
        #vol = zeros((size / 128, size, size, 4), uint32)
        #cu.memcpy_dtoh(vol, gl2cudaMap.device_ptr());

        gl2cudaMap.unmap()
        print "ready"

        #save("vol", vol)
        savez('vvv', mixed = mixedBits.get(), solid = solidBits.get(), debug = debug.get() )

    
    #def display(self):
        #clearGLBuffers()
        #with ctx(self.viewControl.with_vp, self.fragProg(z = self.time*0.1%1.0)):  #self.z)):
        #    drawQuad()

if __name__ == "__main__":
    App().run()
