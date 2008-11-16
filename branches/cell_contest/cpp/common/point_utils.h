#pragma once

#include "geometry/primitives/point.h"

template <class T> inline T min(const cg::point_t<T,3> & p) { return cg::min(p.x, p.y, p.z); }
template <class T> inline T max(const cg::point_t<T,3> & p) { return cg::max(p.x, p.y, p.z); }

template <class T> inline void sort2(cg::point_t<T,3> & a, cg::point_t<T,3> & b)
{
  cg::sort2(a.x, b.x);
  cg::sort2(a.y, b.y);
  cg::sort2(a.z, b.z);
}

template <class T> cg::point_t<T,3> min(const cg::point_t<T,3> & a, const cg::point_t<T,3> & b)
{
  return cg::point_t<T,3> (cg::min(a.x, b.x),  cg::min(a.y, b.y), cg::min(a.z, b.z));
}

template <class T> cg::point_t<T,3> max(const cg::point_t<T,3> & a, const cg::point_t<T,3> & b)
{
  return cg::point_t<T,3> (cg::max(a.x, b.x),  cg::max(a.y, b.y), cg::max(a.z, b.z));
}

template <class T> 
inline int argmin(const cg::point_t<T,3> & p)
{
  int res = 0;
  if (p[1] < p[res]) res = 1;
  if (p[2] < p[res]) res = 2;
  return res;
}

template <class T> 
inline int argmax(const cg::point_t<T,3> & p)
{
  int res = 0;
  if (p[1] > p[res]) res = 1;
  if (p[2] > p[res]) res = 2;
  return res;
}
