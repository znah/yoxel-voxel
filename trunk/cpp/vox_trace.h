#pragma once

#include "trace_cu.h"

#include "geometry/primitives/point.h"
#include "common/point_utils.h"


class TreeTracer
{
private:
  VoxTree m_data;
  uchar4 m_defColor;

  int m_dirFlags;

  uchar4 m_resColor;

  int firstChild(cg::point_3f & t1, cg::point_3f & t2) const 
  {
    int res(0);
    cg::point_3f tm = 0.5f * (t1 + t2);
    float tEnter = max(t1);
    for (int i = 0; i < 3; ++i)
    {
      if (tm[i] > tEnter)
        t2[i] = tm[i];
      else
      {
        t1[i] = tm[i];
        res |= 1<<i;
      }
    }
    return res^m_dirFlags;
  }

  bool nextChild(int & childId, cg::point_3f & t1, cg::point_3f & t2, int exitPlane)
  {
    int mask = 1<<exitPlane;
    if (((childId^m_dirFlags) & mask) != 0)
      return false;

    childId ^= mask;

    float dt = t2[exitPlane] - t1[exitPlane];
    t1[exitPlane] = t2[exitPlane];
    t2[exitPlane] += dt;
    return true;
  }

  bool walkNode(VoxNodeId node, cg::point_3f t1, cg::point_3f t2)
  {
    using namespace cg;

    int childId = firstChild(t1, t2);

    while (true)
    {
      int exitPlane = argmin(t2);

      if (t2[exitPlane] > 0) 
      {
        if ( !IsLeaf(node, childId) )
        {
          if (walkNode(GetChildRef(node, childId), t1, t2))
            return true;
        }
        else
        {
          uchar4 col = GetChildCol(node, childId);
          if (col.w == 255)
          {
            m_resColor = col;
            return true;
          }
        }
      }

      if (!nextChild(childId, t1, t2, exitPlane))
        break;
    }
      
    return false;
  }

  struct SLTrState
  {
    VoxNodeId node;
    int childId;
    cg::point_3f t1;
    cg::point_3f t2;
  };

  bool goDown(SLTrState & state)
  {
    if (IsLeaf(state.node, state.childId))
      return false;

    state.node = GetChildRef(state.node, state.childId);
    state.childId = firstChild(state.t1, state.t2);

    return true;
  }

  bool goNext(SLTrState & state, int exitPlane)
  {
    return nextChild(state.childId, state.t1, state.t2, exitPlane);
  }

  bool goUp(SLTrState & state)
  {
    if (GetParentNode(state.node) < 0)
      return false;

    for (int i = 0; i < 3; ++i)
    {
      float dt = state.t2[i] - state.t1[i];
      int mask = 1<<i;
      (((state.childId^m_dirFlags) & mask) == 0) ? state.t2[i] += dt : state.t1[i] -= dt;
    }
    state.childId = GetSelfChildId(state.node);
    state.node = GetParentNode(state.node);

    return true;
  }

  bool setupTrace(const cg::point_3f & p0, cg::point_3f dir, cg::point_3f & t1, cg::point_3f & t2)
  {
    using namespace cg;

    // avoid speceial case
    float eps = epsilon<float>();
    for (int i = 0; i < 3; ++i)
      if ( abs(dir[i]) < eps )
        dir[i] = dir[i] < 0 ? -eps : eps;

    m_dirFlags = 0;
    for (int i = 0; i < 3; ++i)
      if (dir[i] < 0)
        m_dirFlags |= 1<<i;

    t1 = (point_3f(0, 0, 0) - p0) / dir;
    t2 = (point_3f(1, 1, 1) - p0) / dir;
    sort2(t1, t2);
    float tEnter = max(t1);
    float tExit = min(t2);
    if (tEnter >= tExit || tExit <= 0)
      return false;

    return true;
  }

  uchar4 GetChildCol(VoxNodeId node, uint childId) const { return m_data.children[node*8+childId].color; }
  VoxNodeId GetChildRef(VoxNodeId node, uint childId) const { return m_data.children[node*8+childId].ref; }
  bool IsLeaf(VoxNodeId node, uint childId) const { return (m_data.flags[node].leafFlags & 1<<childId) != 0; }
  uchar GetSelfChildId(VoxNodeId node) const { return m_data.flags[node].selfChildId; }
  VoxNodeId GetParentNode(VoxNodeId node) const { return m_data.parents[node]; }

public:
  TreeTracer(const VoxTree & data) : m_data(data)
  {
    m_defColor = make_uchar4(0, 0, 0, 0);
  }

  void setDefColor(uchar4 color) { m_defColor = color; }

  uchar4 traceRay(const cg::point_3f & p0, const cg::point_3f & dir)
  {
    cg::point_3f t1, t2;
    if (!setupTrace(p0, dir, t1, t2))
      return m_defColor;

    return walkNode(m_data.root, t1, t2) ? m_resColor : m_defColor;
  }

  uchar4 stacklessTrace(const cg::point_3f & p0, const cg::point_3f & dir)
  { 
    SLTrState state;

    if (!setupTrace(p0, dir, state.t1, state.t2))
      return m_defColor;

    state.node = m_data.root;
    state.childId = firstChild(state.t1, state.t2);

    // go down to first intersected leaf
    while (true)
    {
      int exitPlane = argmin(state.t2);
      if (state.t2[exitPlane] <= 0) 
      {
        if (!goNext(state, exitPlane))
          return m_defColor;
      }
      else
      {
        if (!goDown(state))
          break;
      }
    }

    // main trace cycle
    while (true)
    {
      uchar4 col = GetChildCol(state.node, state.childId);
      if (col.w == 255)
        return col;
      
      // go up
      while (!goNext(state, argmin(state.t2)))
        if (!goUp(state))
          return m_defColor;

      while (goDown(state));
    }
  }
};