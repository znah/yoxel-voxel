#pragma once

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>
#include <ctime>

#include "geometry/primitives/point.h"
using cg::point_3i;
using cg::point_3f;
using cg::point_4f;
using cg::point_4b;
using cg::point_4i;
using cg::point_4sb;
using cg::point_2i;

#include "range.h"
#include "grid_walk.h"
#include "array_3d.h"

typedef unsigned char uint8;