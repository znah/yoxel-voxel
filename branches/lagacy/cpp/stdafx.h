#pragma once

#define TARGET_CUDA

#ifdef __CUDACC__
  #define TARGET_CUDA_DEVICE
#else
  #define TARGET_CUDA_HOST
#endif


#if !defined(TARGET_CUDA_DEVICE)
  #define USE_CG
  #define USE_STL
#endif

#ifdef USE_STL
  #include <stdexcept>
  #include <iostream>
  #include <fstream>
  #include <sstream>
  #include <vector>
  #include <algorithm>
  #include <numeric>
  #include <cmath>
  #include <cassert>
  #include <ctime>

  #if __GNUC__ >= 4 
    #include <tr1/memory>
  #else
    #include <boost/shared_ptr.hpp>
  #endif

  #include <boost/noncopyable.hpp>

  #include "format.h"
#endif


#include "cutil_math.h"
#include <cuda_runtime.h>


#ifdef _MSC_VER
  typedef __int64 int64;
  typedef unsigned __int64 uint64;
#else
  typedef long long int64;
  typedef unsigned long long uint64;
#endif

#ifdef USE_CG
  #include "points.h"
  #include "range.h"
  #include "rotation.h"
#endif

/*#ifdef USE_CG
  using cg::min;
  using cg::max;
#elif defined(USE_STL)
  using std::min;
  using std::max;
#endif*/


#ifdef TARGET_CUDA_DEVICE
  #define GLOBAL_FUNC __device__ __host__
#else
  #define GLOBAL_FUNC 
#endif


#ifdef USE_STL
  #if __GNUC__ >= 4
    using std::tr1::shared_ptr;
  #else
    using boost::shared_ptr;
  #endif
#endif

#ifdef USE_STL
  using std::swap;
#else
  template <class T> inline GLOBAL_FUNC void swap(T & a, T & b) { T c = a; a = b; b = c; }
#endif

#ifdef USE_STL
  using boost::noncopyable;
#endif


#ifdef TARGET_CUDA_DEVICE
  #include "cu_cu.h"
  #include "points_cu.h"
#else
  #include "cu_cpp.h"
#endif

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

#ifdef USE_CG
typedef cg::point_4b Color32;
typedef cg::point_4sb Normal32;
#endif

