#pragma once

#include <geometry/primitives/point.h>

struct walk_2
{
  const cg::point_2i size;
  cg::point_2i p;

  int x() const { return p.x; }
  int y() const { return p.y; }
  int flat() const { return p.y*size.x + p.x; }

  walk_2() {}
  walk_2(int sx, int sy) : size(sx, sy) {}
  explicit walk_2(const cg::point_2i & pt) : size(pt) {}

  bool done() const { return p.y >= size.y || p.x >= size.x; }

  walk_2 & operator++() { _next(); return *this; }
  const walk_2 operator++(int) { walk_2 ret = *this; ++(*this); return ret; }

  void _next()
  {
    if (++p.x >= size.x)
    {
      p.x = 0;
      ++p.y;
    }
  }
};


struct walk_3
{
  const cg::point_3i base;
  const cg::point_3i size;
  cg::point_3i p;

  int x() const { return base.x + p.x; }
  int y() const { return base.y + p.y; }
  int z() const { return base.z + p.z; }
  int flat() const { return (z()*size.y + y())*size.x + x(); }

  operator cg::point_3i () const { return pos(); }
  cg::point_3i pos() const { return base + p; }

  walk_3() {}
  walk_3(int sx, int sy, int sz) : size(sx, sy, sz) {}
  explicit walk_3(const cg::point_3i & p) : size(p) {}
  explicit walk_3(int n) : size(n, n, n) {}
  walk_3(const cg::point_3i & p1, const cg::point_3i & p2) : base(p1), size(p2-p1) {}
  walk_3(const range_3i & range) : base(range.p1), size(range.size()) {}
  
  bool done() const { return p.z >= size.z || p.y >= size.y || p.x >= size.x; }
  void next()
  {
    if (++p.x >= size.x)
    {
      p.x = 0;
      if (++p.y >= size.y)
      {
        p.y = 0;
        ++p.z;
      }
    }
  }
};

