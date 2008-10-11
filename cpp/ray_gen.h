#pragma once

#include "geometry/primitives/point.h"
#include "common/grid_walk.h"

class OrthoRayGen
{
private:
  cg::point_3f m_dir;
  cg::point_3f m_eye;
  cg::point_3f m_du, m_dv;
  int m_res;
  walk_2 m_iter;

  cg::point_3f m_p0;

  void calcP0() 
  { 
    m_p0 = m_eye + (m_iter.x()-m_res/2) * m_du + (m_iter.y()-m_res/2) * m_dv; 
  }


public:
  template<class T>
  OrthoRayGen(T dir, int res) : m_iter(res, res)
  {
    m_res = res;
    for (int i = 0; i < 3; ++i)
      m_dir[i] = dir[i];
    normalize(m_dir);
    m_eye = cg::point_3f(0.5, 0.5, 0.5) - m_dir;
    m_du = 2*normalized( m_dir ^ cg::point_3f(0, 0, 1) ) / res * 0.8f;
    m_dv = 2*normalized( m_du ^ m_dir ) / res * 0.8f;
    calcP0();
  }

  bool done() const { return m_iter.done(); }
  void next() { ++m_iter; calcP0(); }

  const cg::point_3f & p0() { return m_p0; }
  const cg::point_3f & dir() { return m_dir; }

  int x() const { return m_iter.x(); }
  int y() const { return m_iter.y(); }
  int flat() const { return m_iter.flat(); }
};