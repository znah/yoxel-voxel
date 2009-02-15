#pragma once

#include "vox_node.h"

struct VoxStructTree
{
  VoxNodeId     root;
  VoxNode     * nodes;
};

struct LightParams
{
  bool   enabled;
  float3 pos;
  float3 diffuse;
  float3 specular;
  float3 attenuationCoefs;
};

const int MaxLightsNum = 3;

struct RayData
{
  VoxNodeId endNode;
  int endNodeChild;
  float endNodeSize;
};

struct RenderParams
{
  int viewWidth;
  int viewHeight;

  float fovCoef; // tan(rp.fov/2)
  float detailCoef;

  float3 eyePos;
  
  matrix_4f viewToWldMtx;
  matrix_4f wldToViewMtx;

  LightParams lights[MaxLightsNum];
  float specularExp;
  float3 ambient;

  RayData * rays;
  float * zBuf;
};

#ifdef __cplusplus
extern "C" {
#endif

void Run_Trace(GridShape grid);
void Run_ShadeSimple(GridShape grid, uchar4 * img);

#ifdef __cplusplus
}
#endif
