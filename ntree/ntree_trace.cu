#include <stdio.h>
#include "cutil_math.h"
#include "cu_cu.h"
#include "ntree_trace.cuh"

texture<uchar4, 3, cudaReadModeNormalizedFloat> voxDataTex;
texture<uint4, 3, cudaReadModeElementType> voxNodeTex;

__constant__ RenderParams rp;

__device__
float3 mul(const float4x4 &M, const float3 &v)
{
    float4 v1 = make_float4(v, 1.0f);
    float3 r;
    r.x = dot(v1, M.m[0]);
    r.y = dot(v1, M.m[1]);
    r.z = dot(v1, M.m[2]);
    return r;
}

#define INIT_THREAD \
  const int xi = blockIdx.x * blockDim.x + threadIdx.x; \
  const int yi = blockIdx.y * blockDim.y + threadIdx.y; \
  const int sx = rp.viewSize.x;                         \
  const int sy = rp.viewSize.y;                         \
  if (xi >= sx || yi >= sy ) return; \
  const int tid = yi*sx + xi;        \


__device__ float3 calcRayDirView(int xi, int yi)
{
  const int sx = rp.viewSize.x;
  const int sy = rp.viewSize.y;
  return make_float3(2*rp.fovCoef*(float)(xi-sx/2)/sx, 2*rp.fovCoef*(float)(yi-sy/2)/sx, -1);
}

__device__ void adjustCoord(float & x)
{
  const float eps = 1e-6f;
  if (fabs(x) < eps)
    x = (x < 0) ? -eps : eps;
}

__device__ void adjustDir(float3 & dir)
{
  adjustCoord(dir.x);
  adjustCoord(dir.y);
  adjustCoord(dir.z);
}

__device__ float3 calcRayDirWorld(int xi, int yi)
{
  float3 dir = calcRayDirView(xi, yi);
  dir = mul(rp.viewToWldMtx, dir);
  dir -= rp.eyePos;
  adjustDir(dir);
  return dir;
}

__device__ float maxCoord(const float3 & p) { return fmaxf(p.x, fmaxf(p.y, p.z)); }
__device__ float minCoord(const float3 & p) { return fminf(p.x, fminf(p.y, p.z)); }

__device__ bool intersectBox(float3 p, float3 invDir, float3 boxmin, float3 boxmax, float3 & t1, float3 & t2)
{
  float3 tbot = invDir * (boxmin - p);
  float3 ttop = invDir * (boxmax - p);
  t1 = fminf(ttop, tbot);
  t2 = fmaxf(ttop, tbot);
  float tnear = maxCoord(t1);
  float tfar  = minCoord(t2);
  return tnear < tfar && tfar > 0;
}

__device__ uint calcDirFlags(float3 dir)
{
  uint res = 0;
  if (dir.x < 0) res |= (ntree::No deSize - 1);
  if (dir.y < 0) res |= (ntree::NodeSize - 1) << ntree::NodeSizePow;
  if (dir.z < 0) res |= (ntree::NodeSize - 1) << (2*ntree::NodeSizePow);
  return res;
}

__global__ void Trace(uchar4 * img, float * debug)
{
  INIT_THREAD;

  float3 dir = calcRayDirWorld(xi, yi);
  float3 invDir = make_float3(1.0f) / dir;
  float3 t1, t2;
  if (!intersectBox(rp.eyePos, invDir, make_float3(0.0f), make_float3(1.0f), t1, t2))
  {
    img[tid] = make_uchar4(0, 0, 0, 255);
    debug[tid] = 0;
    return;
  }
  uint dirFlags = calcDirFlags(dir);

  
  debug[tid] = 0;
  img[tid] = make_uchar4(0, 0, 0, 255);
}


extern "C"
{

  const textureReference * GetDataTex() 
  { 
    voxDataTex.addressMode[0] = cudaAddressModeClamp;
    voxDataTex.addressMode[1] = cudaAddressModeClamp;
    voxDataTex.addressMode[2] = cudaAddressModeClamp;
    voxDataTex.filterMode = cudaFilterModeLinear;
    voxDataTex.normalized = false;

    return &voxDataTex;
  }
  const textureReference * GetNodeTex() 
  { 
    voxNodeTex.addressMode[0] = cudaAddressModeClamp;
    voxNodeTex.addressMode[1] = cudaAddressModeClamp;
    voxNodeTex.addressMode[2] = cudaAddressModeClamp;
    voxNodeTex.filterMode = cudaFilterModePoint;
    voxNodeTex.normalized = false;

    return &voxNodeTex;
  }

  void RunTrace(const RenderParams & params, uchar4 * img, float * debug)
  {
    cudaMemcpyToSymbol(rp, &params, sizeof(params));
    GridShape grid = make_grid2d(params.viewSize, make_int2(16, 16));
    Trace<<<grid.grid, grid.block>>>(img, debug);
    CUT_CHECK_ERROR("Trace");
  }
}