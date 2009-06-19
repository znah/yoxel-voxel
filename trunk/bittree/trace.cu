//
#incldue "cutil_math.h"

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


struct RenderParams
{
  node_id hintTreeRoot;
};

__constant__ RenderParams rp;

texture<uint, 1, cudaReadModeElementType> hint_grid_tex;
texture<uint2, 1, cudaReadModeElementType> hint_brick_tex;



__device__ uint fetchHint(float3 pos)
{
  float nodeSize = 1.0f;
  float3 nodePos = make_float3(0, 0, 0);
  node_id node = rp.hintTreeRoot;
  while ( (node & BrickRefMask) == 0 )
  {
    

  }
  
}


