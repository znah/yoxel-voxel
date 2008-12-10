#pragma once

#include "vox_node.h"

struct VoxStructTree
{
  VoxNodeId     root;
  VoxNode     * nodes;
};

struct RenderParams
{
  int viewWidth;
  int viewHeight;

  float detailCoef;

  point_3f eye;
  point_3f dir;
  point_3f right;
  point_3f up;

  point_3f lightPos;

  float ditherCoef;

  int rndSeed;
};

struct RayData
{
  point_3f pos;
  point_3f dir;
  float t;
  VoxNodeId endNode;
  int endNodeChild;
  float endNodeSize;
};


#ifdef __cplusplus
extern "C" {
#endif

void Run_InitEyeRays(dim3 gridSize, dim3 blockSize, RenderParams rp, RayData * rays, float * noiseBuf);
void Run_Trace(dim3 gridSize, dim3 blockSize, RenderParams rp, RayData * rays);
void Run_ShadeSimple(dim3 gridSize, dim3 blockSize, RenderParams rp, const RayData * eyeRays, uchar4 * img);

#ifdef __cplusplus
}
#endif
