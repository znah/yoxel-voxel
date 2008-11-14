#include "vox_node.h"
#include "rdd.h"

const int MaxRowSize = 1024;

struct trace_spu_params
{
  RayDirData rdd;
  point_2i start, end, viewSize;
  Color32 * colorBuf;

  VoxNodeId root;
  const VoxNode * nodes;
} __attribute__ ((aligned (16)));
