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

struct RenderParams
{
  int viewWidth;
  int viewHeight;

  float detailCoef;

  float3 eyePos;
  float3 dir;
  float3 right;
  float3 up;

  LightParams lights[MaxLightsNum];
  float specularExp;
  float3 ambient;

  float ditherCoef;

  int rndSeed;
  int accumIter;
};


struct RayData
{
  point_3f pos;
  point_3f dir;
  float t;
  VoxNodeId endNode;
  int endNodeChild;
  float endNodeSize;
  int perfCount;
  int unshuffleIndex;
};


#ifdef __cplusplus
extern "C" {
#endif

void Run_InitEyeRays(GridShape grid, RayData * rays, float * noiseBuf, const int * shuffleBuf);
void Run_Trace(GridShape grid, RayData * rays);
void Run_ShadeSimple(GridShape grid, const RayData * eyeRays, uchar4 * img, ushort4 * accum);
void Run_ShadeCounter(GridShape grid, const RayData * eyeRays, uchar4 * img);

#ifdef __cplusplus
}
#endif
