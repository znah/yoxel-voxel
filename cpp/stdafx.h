#pragma once

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;


#ifndef __CUDACC__

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>

#include <cmath>

#endif




#ifndef NO_CUDA
  #include <cuda_runtime.h>
  #include "cutil_math.h"

  #define GLOBAL_FUNC __device__ __host__

#else

  #define GLOBAL_FUNC
#endif


