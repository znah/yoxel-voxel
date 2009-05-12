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