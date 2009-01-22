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
  point_3f pos;
  point_3f dir;
  float t;
  VoxNodeId endNode;
  int endNodeChild;
  float endNodeSize;
};

struct RenderParams
{
  int viewWidth;
  int viewHeight;


  float detailCoef;
  float ditherCoef;

  float3 eyePos;
  float3 dir;
  float3 right;
  float3 up;

  LightParams lights[MaxLightsNum];
  float specularExp;
  float3 ambient;

  int rndSeed;

  RayData * rays;
};

#ifdef __cplusplus
extern "C" {
#endif

void Run_InitEyeRays(GridShape grid, float * noiseBuf);
void Run_Trace(GridShape grid);
void Run_ShadeSimple(GridShape grid, uchar4 * img);
void Run_Blur(GridShape grid, const uchar4 * src, uchar4 * dst);
void Run_BlendLayer(GridShape grid, float t1, float t2, const uchar4 * color, uchar4 * dst);

#ifdef __cplusplus
}
#endif
