#include "vox_node.h"
#include "rdd.h"
#include "shader.h"

const int BlockSize = 16;

struct trace_spu_params
{
  point_3f pos;
  RayDirData rdd;
  point_2i viewSize;
  int blockStart, blockStride;
  Color32 * colorBuf;

  VoxNodeId root;
  const VoxNode * nodes;
} __attribute__ ((aligned (16)));
