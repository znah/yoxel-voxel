#pragma once

#include "common.h"


template<class T, int N>
class multi_array_ref_wrap : public boost::multi_array_ref<T, N>
{
private:
  typedef boost::multi_array_ref<T, N> super;
public:
  multi_array_ref_wrap() : super(NULL, std::vector<int>(N, 0)) {}
  multi_array_ref_wrap(T* base, const int dims[N]) : super(base, std::vector<int>(dims, dims+N)) 
  {
    printf("%d %d\n", dims[0], dims[1]);
  }
};

typedef multi_array_ref_wrap<float, 2> Grid2Dref;
typedef multi_array_ref_wrap<float, 3> V2Grid2Dref;
typedef boost::multi_array<float, 2> Grid2D;
typedef boost::multi_array<float, 3> V2Grid2D;

template <class TGird>
inline bool inside(const TGird & grid, int2 p)
{
    return p.x >= 0 && p.y >= 0 && p.x < grid.shape()[1] && p.y < grid.shape()[0];
}


void calc_distmap(const Grid2Dref & obstmap, Grid2Dref & distmap, V2Grid2Dref & pathmap);
