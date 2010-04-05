//

typedef unsigned int uint;

const int BITVOL_SIZE  = 1024;
const int VOL_SIZE  = BITVOL_SIZE / 2;
const int SLICE_NUM = BITVOL_SIZE / 128;

__device__ uint sumPairs(uint v)
{
  const uint mask = 0x55555555;
  return ((v>>1)&mask) + (v&mask);
}

__device__ uint mergeEvenOdd(uint even, uint odd)
{
  uint res = 0;
  res |=  (even & 0xf)    + ((odd & 0xf)    << 4);          //  0..7
  res |= ((even & 0xf0)   + ((odd & 0xf0)   << 4)) <<  4;   //  8..15
  res |= ((even & 0xf00)  + ((odd & 0xf00)  << 4)) <<  8;   // 16..23
  res |= ((even & 0xf000) + ((odd & 0xf000) << 4)) << 12;   // 24..31
  return res;
}


// block dim = (layer, slice, w) = (4, 8, 8) = 256
extern "C"
__global__ void CalcDensity(const uint* src, uint2 * dst)
{
  int layer = threadIdx.x;
  int slice = threadIdx.y;

  int x = blockIdx.x * blockDim.z + threadIdx.z;
  int y = blockIdx.y;

  uint ofs = (slice*BITVOL_SIZE*BITVOL_SIZE + 2*y*BITVOL_SIZE + 2*x) * 4 + layer;
  uint v0 = sumPairs(src[ofs]);
  uint v1 = sumPairs(src[ofs + 4]);
  uint v2 = sumPairs(src[ofs + BITVOL_SIZE*4]);
  uint v3 = sumPairs(src[ofs + BITVOL_SIZE*4 + 4]);

  const uint mask2 = 0x33333333;
  uint veven = ( v0     & mask2) + ( v1     & mask2) + ( v2     & mask2) + ( v3     & mask2);
  uint vodd  = ((v0>>2) & mask2) + ((v1>>2) & mask2) + ((v2>>2) & mask2) + ((v3>>2) & mask2);

  uint2 res = make_uint2(0, 0);
  res.x = mergeEvenOdd(veven, vodd);
  res.y = mergeEvenOdd(veven>>16, vodd>>16);

  ofs = (y * VOL_SIZE + x) * SLICE_NUM * 4 + slice * 4 + layer;
  dst[ofs] = res; 
}

const int BRICK_SIZE   = 4;
const int COL_WORD_NUM = VOL_SIZE*4/32; // = 64
const int GRID_SIZE    = VOL_SIZE / BRICK_SIZE;
const int OUT_WORD_NUM = GRID_SIZE/32;  // = 4
__shared__ uint s_column[ COL_WORD_NUM+1 ];
__shared__ uint s_empty [ OUT_WORD_NUM ] ;
__shared__ uint s_solid [ OUT_WORD_NUM ];


// block dim = (COL_WORD_NUM, 1, 1)
extern "C"
__global__ void MarkBricks(const uint* g_src, uint * g_mixed, uint * g_solid )
{
  int tid = threadIdx.x;
  if (tid == 0)
    s_column[COL_WORD_NUM] = 0; // pad
  if (tid < OUT_WORD_NUM)
  {
    s_empty[tid] = ~0;
    s_solid[tid] = ~0;
  }

  int bx = blockIdx.x * BRICK_SIZE;
  int by = blockIdx.y * BRICK_SIZE;
  for (int y = by; y <= by + BRICK_SIZE; ++y)
  for (int x = bx; x <= bx + BRICK_SIZE; ++x)
  {
    uint ofs = (x + y * VOL_SIZE) * COL_WORD_NUM + tid;
    uint v = 0;
    if (x < VOL_SIZE && y < VOL_SIZE)
      v = g_src[ofs];
    s_column[tid] = v;
    __syncthreads();

    const uint solidMask = 0x88888;
    const uint emptyMask = 0xFFFFF;
    uint state = 0;
    if ((v & solidMask) == solidMask)
      state |= 1; // solid
    if ((v & emptyMask) == 0)
      state |= 4; // empty

    v = (s_column[tid+1]<<16) | (v>>16);
    if ((v & solidMask) == solidMask)
      state |= 2; // solid
    if ((v & emptyMask) == 0)
      state |= 8; // empty
    
    //if (y < by + BRICK_SIZE && x < bx + BRICK_SIZE)
    //  g_src[ofs] = state; // !!!

    s_column[tid] = state;
    __syncthreads();

    if (tid < OUT_WORD_NUM)
    {
      uint solid = 0;
      uint empty = 0;
      for (int i = 0; i < 16; ++i)
      {
          int o = tid * 16 + i;
          solid |= (s_column[o] & 3)   << (i*2);
          empty |= (s_column[o] & 0xC) << (i*2-2);
      }
      s_empty[tid] &= empty;
      s_solid[tid] &= solid;
    }
  }

  if (tid < OUT_WORD_NUM)
  {
    uint ofs = (blockIdx.x + blockIdx.y * GRID_SIZE) * OUT_WORD_NUM + tid;
    g_mixed[ofs] = ~(s_solid[tid] | s_empty[tid]);
    g_solid[ofs] = s_solid[tid];
  }
}
