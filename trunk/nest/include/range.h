#pragma once

struct range_3i
{
  point_3i p1, p2;

  range_3i() {}
  range_3i(const point_3i & p1_, const point_3i & p2_) : p1(p1_), p2(p2_) 
  {
    for (int i = 0; i < 3; ++i)
      cg::sort2(p1[0], p2[0]);
  }
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

  void operator &=(const range_3i & other)
  {
    for (int i = 0; i < 3; ++i)
    {
      p1[i] = cg::max(p1[i], other.p1[i]);
      p2[i] = cg::min(p2[i], other.p2[i]);
    }
  }
};

inline bool intersect(const range_3i & a, const range_3i & b)
{
  for (int i = 0; i < 3; ++i)
    if (a.p2[i] <= b.p1[i] || b.p2[i] <= a.p1[i] )
      return false;
  return true;
}
