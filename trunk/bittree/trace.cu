//
#include "cutil_math.h"

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
const int BrickSize3 = BrickSize * BrickSize * BrickSize;


#pragma pack(push, 4)
struct RenderParams
{
  node_id hintTreeRoot;
  uint2 viewSize;
};
#pragma pack(pop)

__constant__ RenderParams rp;

texture<uint, 1, cudaReadModeElementType> hint_grid_tex;
texture<uint2, 1, cudaReadModeElementType> hint_brick_tex;


__device__ uint fetchHint(float3 pos)
{
  //float nodeSize = 1.0f;
  float3 nodePos = make_float3(0, 0, 0);
  node_id node = rp.hintTreeRoot;
  while ( (node & BrickRefMask) == 0 )
  {
    pos *= GridSize;
    float3 childPos = floor(pos);
    pos = pos - childPos;
    int childId = (childPos.z * GridSize + childPos.y) + GridSize + childPos.x;
    node = tex1Dfetch(hint_grid_tex, node * GridSize3 + childId);
  }
  if (node == ZeroBlock)
    return 0;
  else if (node == FullBlock)
    return 1;

  pos *= BrickSize;
  float3 voxPos = floor(pos);
  int voxId = (voxPos.z * BrickSize + voxPos.y) + BrickSize + voxPos.x;
  uint2 brick = tex1Dfetch(hint_brick_tex, node * BrickSize3 + voxId);
  uint bits = (voxId < 32) ? brick.x : brick.y;
  return (bits >> (voxId & 0x1f)) & 1;
}




#define INIT_THREAD \
  const int xi = blockIdx.x * blockDim.x + threadIdx.x; \
  const int yi = blockIdx.y * blockDim.y + threadIdx.y; \
  if (xi >= rp.viewSize.x || yi >= rp.viewSize.y ) return; \
  const int tid = yi*rp.viewSize.x + xi;        \


extern "C" {
__global__ void TestFetch(float slice, uint * dst)
{
  INIT_THREAD;
  
  float3 p = make_float3((float)xi / rp.viewSize.x, (float)yi / rp.viewSize.y, slice);
  dst[tid] = fetchHint(p);
}

}


