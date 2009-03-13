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

#include "points.h"
#include "range.h"
#include "common/grid_walk.h"

#include <cuda_runtime.h>

using boost::noncopyable;
using std::swap;


typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

