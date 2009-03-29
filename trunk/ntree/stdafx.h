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

#include <boost/shared_ptr.hpp>

#include <cuda_runtime.h>
#include "cutil_math.h"

#include <boost/noncopyable.hpp>
using boost::noncopyable;


#define GLOBAL_FUNC

#include "points.h"
#include "range.h"
#include "common/grid_walk.h"

#include "matrix.h"
#include "matrix_ops.h"

#include "cu_cpp.h"


using boost::noncopyable;
using std::swap;
using cg::min;
using cg::max;


typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;
