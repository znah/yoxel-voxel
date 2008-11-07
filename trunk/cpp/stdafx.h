#pragma once

#ifdef __CUDACC__
#define TARGET_CUDA
#endif

#if !defined(TARGET_CUDA)
#define USE_STL
#endif

#if !defined(TARGET_CUDA) && !defined(TARGET_PPU) && !defined(TARGET_SPU)
#define USE_CG
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
#endif

#ifdef USE_CG
  #include "point.h"
  #include "range.h"
#endif


#if !defined(TARGET_CUDA)
  #define GLOBAL_FUNC
#else
  #include <cuda_runtime.h>
  #include "cutil_math.h"

  #define GLOBAL_FUNC __device__ __host__
#endif


typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

#ifdef USE_CG
typedef cg::point_4b Color32;
typedef cg::point_4sb Normal32;
#endif
