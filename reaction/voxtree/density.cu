//

#include "cutil_math.h"

typedef unsigned int uint;
typedef unsigned int uint32;
typedef unsigned char uint8;

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
const int GRID_SIZE    = VOL_SIZE / BRICK_SIZE;  // 128
const int COL_WORD_NUM = VOL_SIZE*4/32; // = 64
__shared__ uint s_data[ GRID_SIZE ]; // +1 if COL_WORD_NUM == GRID_SIZE


const uint SOLID_BIT    = 0x80000000;
const uint EMPTY_BIT    = 0x40000000;
const uint UNIFORM_BITS = SOLID_BIT | EMPTY_BIT;

__device__ uint brickState(uint v)
{
  const uint solidMask = 0x88888;
  const uint emptyMask = 0xFFFFF;
  uint state = 0;
  if ((v & solidMask) == solidMask)
    state = SOLID_BIT;
  if ((v & emptyMask) == 0)
    state = EMPTY_BIT;
  return state;
}

__device__ uint prefixSum()
{
  int tid = threadIdx.x;
  const int n = GRID_SIZE;
  int offset = 1;
  for (int d = n>>1; d > 0; d >>= 1)
  {
    if (tid < d)
    {
      int ai = offset * (2*tid+1)-1;
      int bi = offset * (2*tid+2)-1;
      s_data[bi] += s_data[ai];
    }
    offset *= 2;
    __syncthreads();
  }

  uint total = s_data[n-1];
  if (tid == 0) { s_data[n-1] = 0; }

  for (int d = 1; d < n; d *= 2)
  {
    offset >>= 1;
    if (tid < d)
    {
      int ai = offset * (2*tid+1)-1;
      int bi = offset * (2*tid+2)-1;
      uint t = s_data[ai];
      s_data[ai] = s_data[bi];
      s_data[bi] += t;
    }
    __syncthreads();
  }
  return total;
}

// block dim = ( COL_WORD_NUM = GRID_SIZE/2, 1, 1 )
extern "C"
__global__ void MarkBricks(const uint* g_src, uint * g_brickState, uint * g_colsum)
{
  int tid = threadIdx.x;
  if (tid == 0)
    s_data[COL_WORD_NUM] = 0; // pad

  uint state1 = UNIFORM_BITS;
  uint state2 = UNIFORM_BITS;

  int bx = blockIdx.x * BRICK_SIZE;
  int by = blockIdx.y * BRICK_SIZE;
  for (int y = by; y <= by + BRICK_SIZE; ++y)
  for (int x = bx; x <= bx + BRICK_SIZE; ++x)
  {
    uint v = 0;
    uint ofs = (x + y * VOL_SIZE) * COL_WORD_NUM + tid;
    if (x < VOL_SIZE && y < VOL_SIZE)
      v = g_src[ofs];
    s_data[tid] = v;
    __syncthreads();

    state1 &= brickState(v);
    v = (s_data[tid+1]<<16) | (v>>16);
    state2 &= brickState(v);
  }
  s_data[tid*2]   = state1 == 0 ? 1 : 0;
  s_data[tid*2+1] = state2 == 0 ? 1 : 0;
  __syncthreads();

  uint total = prefixSum();
  
  s_data[tid*2]   |= state1;
  s_data[tid*2+1] |= state2;
  __syncthreads();

  int col_id = blockIdx.x + blockIdx.y * GRID_SIZE;
  uint ofs = col_id * GRID_SIZE + tid;
  g_brickState[ofs] = s_data[tid];
  g_brickState[ofs + GRID_SIZE/2] = s_data[tid + GRID_SIZE/2];

  if (tid == 0)
    g_colsum[col_id] = total;
}










__device__ int3 unpackXYZ(uint data)
{
  return make_int3( data & 0xff, (data>>8) & 0xff, (data>>16) & 0xff );
}

__device__ uint packXYZ(int x, int y, int z)
{
  return (z << 16) + (y << 8) + x;
}

__device__ int brickOfs(int x, int y, int z)
{
  return z + x * GRID_SIZE + y * GRID_SIZE * GRID_SIZE;
}

__device__ int brickOfs(int3 brickIdx)
{
  return brickOfs(brickIdx.x, brickIdx.y, brickIdx.z);
}


// block dim = ( GRID_SIZE, 1, 1 )
extern "C"
__global__ void PackBricks(const uint * g_brickData, const uint * g_columnStart, uint * g_out )
{
  int x = blockIdx.x;
  int y = blockIdx.y;
  int z = threadIdx.x;
  uint ofs = brickOfs(x, y, z);
  uint data = g_brickData[ofs];
  if ((data & UNIFORM_BITS) == 0)
  {
    uint outidx = g_columnStart[x + y * GRID_SIZE] + data;
    g_out[outidx] = packXYZ(x, y, z);
  }
}





// mapped enum -> packed bricks -> volume -> mapped slots


const int MAX_MAPPED_SLOTS = 16;
__constant__ int slot2slice[MAX_MAPPED_SLOTS];

const int BRICK_SIZE_PAD = BRICK_SIZE+1;
const int BRICK_SIZE_PAD2 = BRICK_SIZE_PAD * BRICK_SIZE_PAD;
const int BRICK_SIZE_PAD3 = BRICK_SIZE_PAD2 * BRICK_SIZE_PAD;

//__shared__ uint s_brickvol[BRICK_SIZE+1];


// block size: brickSize^3,  grid size: sliceSizeX*slotNum, sliceSizeY
extern "C"
__global__ void UpdateBrickPool(
  int sliceSizeX,
  const int  * g_mappedEnum, 
  const uint * g_packedBrickPos, 
  const uint * g_volume, 
  uint8 * g_mappedBricks,
  uint * g_brickState
)
{
  int sliceSizeY = gridDim.y;
  int mapSlot = blockIdx.x / sliceSizeX;
  int mapX = blockIdx.x % sliceSizeX;
  int mapY = blockIdx.y;
  int mapIdx = blockIdx.x + blockIdx.y * gridDim.x;

  int packedIdx = g_mappedEnum[mapIdx];
  if (packedIdx < 0)
    return;

  int3 brickpos = unpackXYZ( g_packedBrickPos[packedIdx] );
  int3 pos = brickpos * BRICK_SIZE;
  pos.x += threadIdx.x;
  pos.y += threadIdx.y;
  pos.z += threadIdx.z;
  if (threadIdx.x >= BRICK_SIZE || threadIdx.y >= BRICK_SIZE || threadIdx.z >= BRICK_SIZE)
    return;

  int volofs = (pos.z/8) * VOL_SIZE * VOL_SIZE + pos.y * VOL_SIZE + pos.x;
  uint d = g_volume[volofs];
  uint v = (d >> ((pos.z%8) * 4)) & 0xf;

  int tid = threadIdx.x + (threadIdx.y + threadIdx.z * blockDim.y) * blockDim.x;
  
  mapX = mapX * BRICK_SIZE_PAD + threadIdx.x;
  mapY = mapY * BRICK_SIZE_PAD + threadIdx.y;
  int layerStride = sliceSizeY * sliceSizeY * BRICK_SIZE_PAD2;
  int ofs = layerStride * (threadIdx.z + mapSlot * BRICK_SIZE_PAD) + mapY * sliceSizeX * BRICK_SIZE_PAD + mapX;
  g_mappedBricks[ofs] = v;


  if (tid == 0)
    g_brickState[ brickOfs(brickpos) ] = packXYZ(mapX, mapY, slot2slice[mapSlot]);
}







texture<uint8, 3, cudaReadModeNormalizedFloat> brick_pool_tex;
texture<uint32, 1, cudaReadModeElementType>    brick_grid_tex;

__device__ uchar4 VolFetch(float3 globalPos)
{
  float3 brickPos = globalPos * (1.0f / BRICK_SIZE);
  int3   brickIdx = make_int3(brickPos);
  float3 localPos = (brickPos - make_float3(brickIdx)) * BRICK_SIZE;

  uint32 brickData = tex1Dfetch(brick_grid_tex, brickOfs(brickIdx));
  
  if (brickData & SOLID_BIT)
    return make_uchar4(0, 0, 128, 0);
  if (brickData & EMPTY_BIT)
    return make_uchar4(0, 0, 0, 0);
  
  int3   poolIdx  = unpackXYZ(brickData);
  float3 poolPos = make_float3(poolIdx) * BRICK_SIZE_PAD + make_float3(0.5f) + localPos;
  float v = tex3D(brick_pool_tex, poolPos.x, poolPos.y, poolPos.z) * (255.0f / 8.0f);

  return make_uchar4(64, v*255, 0, 1);
}


extern "C"
__global__ void FetchTest(float x0, float y0, float z0, uchar4 * g_out)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  const int viewWidth = blockDim.x * gridDim.x; 

  float3 p = make_float3(x0 + x*0.5, y0 + y*0.5, z0);
  g_out[x + y * viewWidth] = VolFetch(p);
}


