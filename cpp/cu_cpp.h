#pragma once

#include "cu_cu.h"

// TODO
#define CUDA_SAFE_CALL 

#  define CUT_CHECK_ERROR(errorMessage) do {                                 \
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


template <class T>
class CuVector : public noncopyable
{
private:
  T * d_data;
  size_t m_size;

  void _alloc(size_t size)
  {
    CUDA_SAFE_CALL( cudaMalloc((void**)&d_data, sizeof(T) * size) );
    m_size = size;
  }
  void _free()
  {
    if (m_size == 0)
      return;
    CUDA_SAFE_CALL( cudaFree(d_data) );
    d_data = 0;
    m_size = 0;
  }

public:
  CuVector() : d_data(0), m_size(0) {}
  CuVector(size_t size) : d_data(0), m_size(0) { _alloc(size); }
  CuVector(size_t size, const T * data) : d_data(0), m_size(0)
  {
    _alloc(size);
    write(0, m_size, data);
  }

  CuVector(const std::vector<T> & vec) : d_data(0), m_size(0)
  {
    _alloc(vec.size());
    write(0, m_size, &vec[0]);
  }

  virtual ~CuVector() { _free(); }

  void resize(size_t size) 
  { 
    if (m_size == size) 
      return;
    _free(); _alloc(size); 
  }

  size_t size() const { return m_size; }

  void write(size_t start, size_t n, const T * src) 
  {
    assert(start + n <= m_size);
    CUDA_SAFE_CALL( cudaMemcpy(d_data + start, src, n*sizeof(T), cudaMemcpyHostToDevice) );
  }

  void read(size_t start, size_t n, T * dst)
  {
    assert(start + n <= m_size);
    CUDA_SAFE_CALL( cudaMemcpy(dst, d_data + start, n*sizeof(T), cudaMemcpyDeviceToHost) );
  }

  void read(std::vector<T> & dst)
  {
    dst.resize(m_size);
    read(0, m_size, &dst[0]);
  }

//  operator T* () { return d_data; }
  T* d_ptr() { return d_data; }
};


template<class T>
void CuSetSymbol(const T & src, const char * dest)
{
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(dest, (const void *) &src, sizeof(T)) );
}

template<class T>
void CuGetSymbol(const char * src, T & dest)
{
  CUDA_SAFE_CALL( cudaMemcpyFromSymbol((void *)dest, src, sizeof(T)) );
}

inline float3 make_float3(const point_3f & p) { return make_float3(p.x, p.y, p.z); }

inline int iDivUp(int a, int b)
{
  int r = a / b;
  return (r*b < a) ? r+1 : r;
}

inline GridShape make_grid2d(const point_2i & size, const point_2i & block)
{
  GridShape shape;
  shape.block = dim3(block.x, block.y, 1);
  shape.grid = dim3(iDivUp(size.x, block.x), iDivUp(size.y, block.y), 1);
  return shape;
}
