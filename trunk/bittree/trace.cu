//
#include "cutil_math.h"
#include "cu_matrix.h"

// 31 bit 
// - grid(0)
// - brick(1)
// special:
// 0xFF FF FF F0 - all zero
// 0xFF FF FF F1 - all one

typedef uint node_id;

const node_id ZeroBlock    = 0xFFFFFFF0;
const node_id FullBlock    = 0xFFFFFFF1;
const node_id BrickRefMask = 0x80000000;

const int BrickSize = 4;
const int GridSize = 4;

const int GridSize3 = GridSize * GridSize * GridSize;


#pragma pack(push, 4)
struct RenderParams
{
  node_id hintTreeRoot;
  uint2 viewSize;

  float fovCoef; // tan(fov/2)

  float3 eyePos;
  float3x4 viewToWldMtx;
  float3x4 wldToViewMtx;
};
#pragma pack(pop)

__constant__ RenderParams rp;

texture<uint, 1, cudaReadModeElementType> hint_grid_tex;
texture<uint2, 1, cudaReadModeElementType> hint_brick_tex;




__device__ uint fetchHint(float3 pos)
{
  float3 nodePos = make_float3(0, 0, 0);
  node_id node = rp.hintTreeRoot;
  while ( (node & BrickRefMask) == 0 )
  {
    pos *= GridSize;
    float3 childPos = floor(pos);
    pos = pos - childPos;
    int childId = (childPos.z * GridSize + childPos.y) * GridSize + childPos.x;
    node = tex1Dfetch(hint_grid_tex, node * GridSize3 + childId);
  }
  if (node == ZeroBlock)
    return 0;
  else if (node == FullBlock)
    return 1;
  
  node &= ~BrickRefMask;
  pos *= BrickSize;
  float3 voxPos = floor(pos);
  int voxId = (voxPos.z * BrickSize + voxPos.y) * BrickSize + voxPos.x;
  uint2 brick = tex1Dfetch(hint_brick_tex, node);
  uint bits = (voxId < 32) ? brick.x : brick.y;
  return (bits >> (voxId & 0x1f)) & 1;
}


__device__ float3 CalcRayViewDir(int xi, int yi)
{
  const int sx = rp.viewSize.x;
  const int sy = rp.viewSize.y;
  float dl = 2.0f * rp.fovCoef / sx;
  return make_float3( dl*(xi-sx/2), dl*(yi-sy/2), -1 );
}

__device__ float3 CalcRayWorldDir(int xi, int yi)
{
  float3 dir = CalcRayViewDir(xi, yi);
  dir = mul(rp.viewToWldMtx, dir);
  return dir;
}
__device__ int sign(float v)
{
  return v > 0 ? 1 : (v < 0 ? -1 : 0);
}

__device__ bool hitBox(float3 dir, float3 orig, float3 boxMin, float3 boxMax, float3 & t1, float3 & t2)
{
  float3 invDir = make_float3(1.0f) / dir;
  float3 tt1 = invDir * (boxMin - orig);
  float3 tt2 = invDir * (boxMax - orig);
  t1 = fminf(tt1, tt2);
  t2 = fmaxf(tt1, tt2);
  float tenter = fmaxf( fmaxf(t1.x, t1.y), t1.z );
  float texit  = fminf( fminf(t2.x, t2.y), t2.z );

  return (texit > 0.0f) && (tenter < texit);
}






#define INIT_THREAD \
  const int xi = blockIdx.x * blockDim.x + threadIdx.x; \
  const int yi = blockIdx.y * blockDim.y + threadIdx.y; \
  if (xi >= rp.viewSize.x || yi >= rp.viewSize.y ) return; \
  const int tid = yi*rp.viewSize.x + xi;        \


extern "C" {

__global__ void TestFetch(float slice, float * dst)
{
  INIT_THREAD;
  
  float3 p = make_float3((float)xi / rp.viewSize.x, (float)yi / rp.viewSize.y, slice);
  dst[tid] = fetchHint(p);
}

__global__ void Trace(float * dst)
{
  INIT_THREAD;
  float3 dir = CalcRayWorldDir(xi, yi);
  int3 dirBits = make_int3( signbit(dir.x), signbit(dir.y), signbit(dir.z) );
  float3 t1, t2;
  if (!hitBox(dir, rp.eyePos, make_float3(0.0f), make_float3(1.0f), t1, t2))
  {
    dst[tid] = 0.0f;
    return;
  }






  float dzx = abs(dir.z / dir.x);
  float dyx = abs(dir.y / dir.x);
  float dzy = abs(dir.z / dir.y);
  float ezx = t2.x * abs(dir.z) + rp.eyePos.z;
  float eyx = t2.x * abs(dir.y) + rp.eyePos.y;
  float ezy = t2.y * abs(dir.z) + rp.eyePos.z;
                     

  dst[tid] = ezx;
  
}
 
}


