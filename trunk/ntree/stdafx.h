#pragma once

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

#include <cuda_runtime.h>
#include "cutil_math.h"


#define GLOBAL_FUNC

#include "points.h"
#include "range.h"
#include "common/grid_walk.h"

#include "matrix.h"
#include "matrix_ops.h"


using boost::noncopyable;
using std::swap;
using cg::min;
using cg::max;


typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;
