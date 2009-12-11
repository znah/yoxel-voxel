#pragma once

namespace zg {

template <class T>
struct point_2t;

typedef point_2t<int> point_2i;
typedef point_2t<float> point_2f;

template <class T>
struct point_2t
{
  T x, y;
  point_2t() : x(0), y(0) {}
  point_2t(T x, T y) : x(x), y(y) {}
};

}