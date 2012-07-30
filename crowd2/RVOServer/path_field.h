#pragma once

#include "common.h"

template<class T>
class array_2d_ref
{
public:
   enum {NDim = 2};
   T * data;
   int shape[2];
   array_2d_ref() : data(NULL) 
   { 
     std::fill(shape, shape+NDim, 0);
   }
   array_2d_ref(int shape_[NDim], T* data_) : data(data_)
   {
     for (int i = 0; i < NDim; ++i)
       shape[i] = shape_[i];
   }

   T& operator()(int i, int j) { return data[i*shape[1] + j]; }
   const T& operator()(int i, int j) const { return data[i*shape[1] + j]; }

   T& operator()(int2 p) { return data[p.y*shape[1] + p.x]; }
   const T& operator()(int2 p) const { return data[p.y*shape[1] + p.x]; }

   bool inside(int2 p) const 
   { return p.x >= 0 && p.y >= 0 && p.x < shape[1] && p.y < shape[0]; }
};


typedef array_2d_ref<float> Grid2Dref;
typedef array_2d_ref<float2> V2Grid2Dref;


void calc_distmap(const Grid2Dref & obstmap, Grid2Dref & distmap, V2Grid2Dref & pathmap);
