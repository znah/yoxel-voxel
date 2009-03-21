#pragma once


struct float4x4
{
  float4 m[4];
};

struct RenderParams
{
  int2 viewSize;

  float fovCoef; // tan(fov/2)
  float pixelAng; // fov / viewWidth

  float3 eyePos;
  
  float4x4 viewToWldMtx;
  float4x4 wldToViewMtx;
};


extern "C"
{
  const textureReference * GetDataTex();
  const textureReference * GetNodeTex();

  void RunTrace(const RenderParams & params, uchar4 * img);
}