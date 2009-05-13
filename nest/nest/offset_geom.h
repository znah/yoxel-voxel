#pragma once

namespace offset_geom
{

struct Sphere
{
  float radius;

  explicit Sphere(float r = 1.0f) : radius(r) {}

  float operator()(const point_3f & p) const
  {
    return radius - cg::norm(p);
  };
};

struct Rect
{
  point_3f halfSize;

  explicit Rect(const point_3f & size_ = point_3f(1, 1, 1)) : halfSize(0.5f * size_) {}

  float operator()(const point_3f & p) const
  {
    return 0; // TODO
     
  }

};

template <class Base>
struct Translate
{
  Base base;
  cg::point_3f ofs;
  Translate() {}
  explicit Translate(const Base & base_) : base(base_) {}
  Translate(const Base & base_, const point_3f & ofs_) : base(base_), ofs(ofs_) {}

  float operator()(const point_3f & p) const
  {
    return base(p - ofs);
  }
};

}