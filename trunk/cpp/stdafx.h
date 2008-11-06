#pragma once


#ifndef __CUDACC__

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>

#include <cmath>

#include "point.h"
#include "range.h"

#endif


#ifndef NO_CUDA
  #include <cuda_runtime.h>
  #include "cutil_math.h"

  #define GLOBAL_FUNC __device__ __host__
#else

  #define GLOBAL_FUNC
#endif


typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

#ifndef __CUDACC__
typedef cg::point_4b Color32;
typedef cg::point_4sb Normal32;
#endif