#pragma once

#ifdef __CUDACC__
#define TARGET_CUDA
#endif

#if defined(TARGET_PPU) || defined(TARGET_SPU)
  #define TARGET_CELL
#else
  #define TARGET_GPU
#endif

#if !defined(TARGET_CUDA)// && !defined(TARGET_PPU) && !defined(TARGET_SPU)
#define USE_CG
#endif

#if !defined(TARGET_CUDA) && !defined(TARGET_SPU)
#define USE_STL
#endif

#ifdef TARGET_SPU
  #include <stdio.h>
  #include <stdlib.h>
  #include <spu_intrinsics.h>
  
  extern "C" {
  #include <spu_mfcio.h>
  }
#endif


#ifdef USE_STL
  #include <stdexcept>
  #include <iostream>
  #include <fstream>
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
#endif

#ifdef __GNUC__
  #include <sys/time.h>
#endif

#ifdef TARGET_GPU
  #include <cuda_runtime.h>
#endif

#if !defined(TARGET_CUDA)
  #define GLOBAL_FUNC
#else
  #include <cuda_runtime.h>
  #include "cutil_math.h"

  #define GLOBAL_FUNC __device__ __host__
#endif


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

#ifdef USE_CG
  using cg::min;
  using cg::max;
#elif defined(USE_STL)
  using std::min;
  using std::max;
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

#ifdef TARGET_CUDA
  #include "points_cu.h"
#endif


typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

#ifdef USE_CG
typedef cg::point_4b Color32;
typedef cg::point_4sb Normal32;
#endif
