#pragma once

#include "common.h"

typedef boost::multi_array_ref<float, 2> Grid2Dref;
typedef boost::multi_array_ref<float, 3> V2Grid2Dref;
typedef boost::multi_array<float, 2> Grid2D;
typedef boost::multi_array<float, 3> V2Grid2D;

template <class TGird>
inline bool inside(const TGird & grid, int2 p)
{
    return p.x >= 0 && p.y >= 0 && p.x < grid.shape()[1] && p.y < grid.shape()[0];
}


void calc_distmap(const Grid2Dref & obstmap, Grid2Dref & distmap, V2Grid2Dref & pathmap);
