#pragma once

#include "ntree/gpu_nodes.h"

struct RenderParams
{
  int2 viewSize;

  float fovCoef; // tan(fov/2)
  float pixelAng; // fov / viewWidth

  float3 eyePos;
  
  float4x4 viewToWldMtx;
  float4x4 wldToViewMtx;

  GPURef rootGrid;
  GPURef rootData;
};


extern "C"
{
  const textureReference * GetDataTex();
  const textureReference * GetNodeTex();

  void RunTrace(const RenderParams & params, uchar4 * img, float * debug);
}