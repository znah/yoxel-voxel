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

  void trace(point_3f p, point_3f dir)
  {
    hit = false;
    adjustDir(dir);
    dxy = point_2f




    
    

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