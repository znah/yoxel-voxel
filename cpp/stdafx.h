#pragma once

#ifdef __CUDACC__
#define TARGET_CUDA
#ednif


#if !defined(TARGET_CUDA)
  #include <stdexcept>
  #include <iostream>
  #include <fstream>
  #include <vector>
  #include <algorithm>
  #include <numeric>

  #include <cmath>

  #include "point.h"
  #include "range.h"

  #define GLOBAL_FUNC
#else
  #include <cuda_runtime.h>
  #include "cutil_math.h"

  #define GLOBAL_FUNC __device__ __host__
#endif



typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

#ifndef TARGET_CUDA
typedef cg::point_4b Color32;
typedef cg::point_4sb Normal32;
#endif