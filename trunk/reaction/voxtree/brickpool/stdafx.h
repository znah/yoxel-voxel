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
#include <cmath>
#include <map>
#include <set>
#include <hash_map>

#include <cuda.h>


#pragma warning( push )
  #pragma warning (disable : 4267 4244 )
  
  #include <boost/python.hpp>

#pragma warning( pop )

#include <boost/noncopyable.hpp>

#pragma warning (disable : 4018) // C4018: '<' : signed/unsigned mismatch

/*
#include "nvVector.h"
typedef nv::vec3<float> point_3f;
typedef nv::vec2<float> point_2f;
*/

using boost::noncopyable;

typedef unsigned int  uint32;
typedef unsigned char uint8;
