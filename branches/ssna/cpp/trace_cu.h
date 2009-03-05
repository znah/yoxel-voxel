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

  float fovCoef; // tan(fov/2)
  float pixelAng; // fov / viewWidth
  float detailCoef;

  float3 eyePos;
  
  matrix_4f viewToWldMtx;
  matrix_4f wldToViewMtx;

  LightParams lights[MaxLightsNum];
  float specularExp;
  float3 ambient;

  RayData * rays;
  float * zbuf;

  bool ssna;
  bool showNormals;
};

const int BlurZKernSize = 7;
const int NoiseBufSize = 256;

#ifdef __cplusplus
extern "C" {
#endif

void Run_Trace(GridShape grid);
void Run_ShadeSimple(GridShape grid, uchar4 * img, const float * zbuf);
void Run_BlurZ(GridShape grid, float farLimit, const float * src, float * dst);

#ifdef __cplusplus
}
#endif
