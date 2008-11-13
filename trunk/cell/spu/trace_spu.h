#include "vox_node.h"

struct trace_spu_params
{
  RayDirData rdd;
  point_2i start, end, viewSize;
  Color32 * colorBuf;

  VoxNodeId root;
  const VoxNode * nodes;
};

extern spe_program_handle_t trace_spu;