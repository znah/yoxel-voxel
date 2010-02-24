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

#pragma warning( push )
  #pragma warning (disable : 4267 4244 )
  
  #include <boost/python.hpp>

#pragma warning( pop )

#pragma warning (disable : 4018) // C4018: '<' : signed/unsigned mismatch

#include "nvVector.h"
typedef nv::vec3<float> point_3f;
typedef nv::vec2<float> point_2f;

