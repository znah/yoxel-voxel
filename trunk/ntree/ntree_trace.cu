#include <stdio.h>
#include "cutil_math.h"
#include "ntree_trace.cuh"

texture<uchar4, 3, cudaReadModeNormalizedFloat> voxDataTex;
texture<uint4, 3, cudaReadModeElementType> voxChildTex;

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


inline int divUp(int a, int b)
{
  return (a+b-1) / b;
}

struct GridShape
{
  dim3 grid;
  dim3 block;
};

inline GridShape make_grid2d(const int2 & size, const int2 & block)
{
  GridShape shape;
  shape.block = dim3(block.x, block.y, 1);
  shape.grid = dim3(divUp(size.x, block.x), divUp(size.y, block.y), 1);
  return shape;
}

#define CUT_CHECK_ERROR(errorMessage) do {                                 \
  cudaError_t err = cudaGetLastError();                                    \
  if( cudaSuccess != err) {                                                \
      fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
              errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
      exit(EXIT_FAILURE);                                                  \
  }                                                                        \
  err = cudaThreadSynchronize();                                           \
  if( cudaSuccess != err) {                                                \
      fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
              errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
      exit(EXIT_FAILURE);                                                  \
  } } while (0)

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
    voxChildTex.addressMode[0] = cudaAddressModeClamp;
    voxChildTex.addressMode[1] = cudaAddressModeClamp;
    voxChildTex.addressMode[2] = cudaAddressModeClamp;
    voxChildTex.filterMode = cudaFilterModePoint;
    voxDataTex.normalized = false;
    return &voxChildTex; 
  }

  void RunTrace(const RenderParams & params, uchar4 * img)
  {
    cudaMemcpyToSymbol(rp, &params, sizeof(params));
    GridShape grid = make_grid2d(params.viewSize, make_int2(16, 16));
    Trace<<<grid.grid, grid.block>>>(img);
    CUT_CHECK_ERROR("Trace");
  }
}