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


__global__ void Trace(uchar4 * img)
{
  INIT_THREAD;

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

    //const textureReference * texref(NULL);
    //CUDA_SAFE_CALL( cudaGetTextureReference(&texref, "voxDataTex") );
    //return texref; 
    return &voxDataTex;
  }
  const textureReference * GetNodeTex() 
  { 
    voxNodeTex.addressMode[0] = cudaAddressModeClamp;
    voxNodeTex.addressMode[1] = cudaAddressModeClamp;
    voxNodeTex.addressMode[2] = cudaAddressModeClamp;
    voxNodeTex.filterMode = cudaFilterModePoint;
    voxNodeTex.normalized = false;

    //const textureReference * texref;
    //CUDA_SAFE_CALL( cudaGetTextureReference(&texref, "voxChildTex") );
    //return texref; 
    return &voxNodeTex;
  }

  void RunTrace(const RenderParams & params, uchar4 * img)
  {
    cudaMemcpyToSymbol(rp, &params, sizeof(params));
    GridShape grid = make_grid2d(params.viewSize, make_int2(16, 16));
    Trace<<<grid.grid, grid.block>>>(img);
    CUT_CHECK_ERROR("Trace");
  }
}