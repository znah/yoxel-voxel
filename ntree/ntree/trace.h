#pragma once
#include "nodes.h"

namespace ntree
{

struct RayTracer
{
  const float dirEps;

  int isoLevel;
  const Node * root;

  RayTracer(int isoLevel_, const Node * root_)
  : dirEps(1e-5)
  , isoLevel(isoLevel_)
  , root(root_)
  {}

  // result
  bool hit;
  cg::point_3f pos;
  cg::point_3f normal;
  cg::point_3f color;

  // local
  point_3i dirSign;
  point_2f dxy, dyz, dzx; 
  struct Section
  {
    int plane;
    point_2f xy, yz, zx;
  };

  void trace(point_3f p, point_3f dir)
  {
    hit = false;
    adjustDir(dir);
    dxy = point_2f(dir.x, dir.y) / dir.z;
    dyz = point_2f(dir.y, dir.z) / dir.x;
    dzx = point_2f(dir.z, dir.x) / dir.y;
    Section s;
    point_3f enterPlanes = 0.5f * (point_3f(1, 1, 1) - dirSign);
    s.xy = p.xy + dxy * (enterPlanes.z - p.z);
    s.yz = p.yz + dyz * (enterPlanes.x - p.x);
    s.zx = p.zx + dzx * (enterPlanes.y - p.y);
    s.plane = findPlane(s);
    hit = traceNode(root, s, 1, point_3i(0, 0, 0));
  }
  
  bool traceNode(Node * node, Section s, int levelSize, point_3i levelPos)
  {
    int nodeSize = (node->GetType() == Node::Brick) ? ntree::BrickSize : ntree::GridSize;
    s.xy *= nodeSize;
    s.yz *= nodeSize;
    s.zx *= nodeSize;
    point_3i child;
    if (s.plane == 2)
    {
      child.x = cg::floor(s.xy[0]);
      child.y = cg::floor(s.xy[1]);
      child.z = dirSign.z > 0 : 0 ? nodeSize-1;

    }
    else if (s.plane == 0)
    {
      child.x = dirSign.x > 0 : 0 ? nodeSize-1;
      child.y = cg::floor(s.yz[0]);
      child.z = cg::floor(s.yz[1]);

    }
    else if (s.plane == 1)
    {
      child.x = cg::floor(s.yz[0]);
      child.y = dirSign.y > 0 : 0 ? nodeSize-1;
      child.z = cg::floor(s.yz[1]);

    }
    else
      return false;


  }

  bool inFace(const point_2f & p)
  {
    return p.x >= 0 && p.y >=0 && p.x < 1 && p.y < 1;
  }

  int findPlane(const Section & s)
  {
    if (inFace(s.xy)) return 2;
    if (inFace(s.yz)) return 0;
    if (inFace(s.zx)) return 1;
    return -1;
  }

  void adjustDir(point_3f & dir)
  {
    for (int i = 0; i < 3; +i)
    {
      float x = dir[i];
      dirSign[i] = cg::sign(x);
      if (abs(x) < dirEps)
        x = cg::sign(x) * dirEps;
      dir[i] = x;
    }
  }
};





}