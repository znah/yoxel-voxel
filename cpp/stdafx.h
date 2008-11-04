#pragma once


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


typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;


#pragma pack(push, 1)

struct Color32
{
  uchar r, g, b, a;

  Color32() : r(0), g(0), b(0), a(0) {}
  Color32(uchar r_, uchar g_, uchar b_, uchar a_ = 0) : r(r_), g(g_), b(b_), a(a_) {}
};

struct Normal32
{
  char x, y, z, w;



};

#pragma pack(pop)
