#pragma once

struct GridShape
{
  dim3 grid;
  dim3 block;
};

inline int divUp(int a, int b)
{
  return (a+b-1) / b;
}

inline GridShape make_grid2d(const int2 & size, const int2 & block)
{
  GridShape shape;
  shape.block = dim3(block.x, block.y, 1);
  shape.grid = dim3(divUp(size.x, block.x), divUp(size.y, block.y), 1);
  return shape;
}


struct float4x4
{
  float4 m[4];
};

#  define CUDA_SAFE_CALL_NO_SYNC( call) do {                                 \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        /*exit(EXIT_FAILURE);*/                                                  \
    } } while (0)

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);                                            \


#  define CUT_CHECK_ERROR(errorMessage) do {                                 \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        /*exit(EXIT_FAILURE);*/                                                  \
    }                                                                        \
    err = cudaThreadSynchronize();                                           \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        /*exit(EXIT_FAILURE);*/                                                  \
    } } while (0)
