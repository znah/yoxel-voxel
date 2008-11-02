#pragma once

#include "point.h"

struct range_3i
{
  point_3i p1, p2;

  range_3i() {}
  range_3i(const point_3i & p1_, const point_3i & size) : p1(p1_), p2(p1_ + size) {}
  range_3i(const point_3i & p1_, int size) : p1(p1_), p2(p1_ + point_3i(size, size, size)) {}

  point_3i size() const { return p2-p1; }

  bool empty() 
  {  
    for (int i = 0; i < 3; ++i)
      if (p2[i] <= p1[i])
        return true;
    return false;
  }

  bool contains(const range_3i & other) const
  {
    for (int i = 0; i < 3; ++i)
      if (p1[i] > other.p1[i] || other.p2[i] > p2[i])
        return false;
    return true;
  }

  bool intersects(const range_3i & other) const
  {
    for (int i = 0; i < 3; ++i)
      if (p2[i] <= other.p1[i] || other.p2[i] <= p1[i] )
        return false;
    return true;
  }

  void operator &=(const range_3i & other)
  {
    for (int i = 0; i < 3; ++i)
    {
      p1[i] = cg::max(p1[i], other.p1[i]);
      p2[i] = cg::min(p2[i], other.p2[i]);
    }
  }
};
