#pragma once

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>
#include <ctime>

#pragma warning( push )
  #pragma warning (disable : 4267 4244 )
  
  #include <boost/python.hpp>
#pragma warning( pop )

#include "nvVector.h"

typedef nv::vec2<float> point_2f;
typedef nv::vec2<int> point_2i;
typedef nv::vec3<float> point_3f;
typedef nv::vec3<int> point_3i;

#pragma warning (disable : 4018) // C4018: '<' : signed/unsigned mismatch

const float epsf = 1e-5f;

inline float min(float a, float b) { return a < b ? a : b; }
inline float max(float a, float b) { return a > b ? a : b; }

inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }

inline float randf() { return (float)rand() / RAND_MAX; }
