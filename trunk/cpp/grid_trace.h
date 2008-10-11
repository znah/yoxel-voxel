#pragma once

#include <algorithm>
#include "common/point_utils.h"

class grid3_tracer
{
public:
  grid3_tracer(int n, cg::point_3f p0, cg::point_3f dir)
  {
    using namespace cg;
    // convert to grid coordinates
    m_gridN = n;
    p0 *= n;
    dir *= n;
    
    point_3f dt = point_3f(1, 1, 1) / dir;
    point_3f t1 = -p0 & dt;
    point_3f t2 = (point_3f(n, n, n)-p0) & dt;
    sort2(t1, t2);
    
    float tEnter = max(t1);
    float tExit = min(t2);

    if (tEnter >= tExit || tExit <= 0)
    {
      // no intersection with grid
      m_done = true;
      return;
    }
    
    m_curT = max(tEnter, 0.0f);
    m_idx = point_3i(p0 + dir * m_curT);
    for (int i = 0; i < 3; ++i)
      m_idx[i] = limit(0, n-1)(m_idx[i]);
    
    t1 = (m_idx-p0) & dt;
    t2 = (m_idx+point_3f(1, 1, 1)-p0) & dt;
    sort2(t1, t2);
    m_nextT = t2;
    
    m_dt = point_3f(abs(dt.x), abs(dt.y), abs(dt.z));
    m_gridDir = point_3i(sign(dir.x), sign(dir.y), sign(dir.z));
    
    m_done = false;
  }

  const cg::point_3i & p() const { return m_idx; }
  float t() const { return m_curT; }
  bool done() const { return m_done; }

  void next() 
  {
    if (m_done)
      return;
    int i = 0;
    if (m_nextT[1] < m_nextT[i]) i = 1;
    if (m_nextT[2] < m_nextT[i]) i = 2;
    m_idx[i] += m_gridDir[i];
    if (m_idx[i] < 0 || m_idx[i] >= m_gridN)
    {
      m_done = true;
      return;
    }
    m_curT = m_nextT[i];
    m_nextT[i] += m_dt[i];
  }

private:
  bool m_done;
  int m_gridN;
  cg::point_3i m_idx, m_gridDir;
  float m_curT;
  cg::point_3f m_nextT, m_dt;
};