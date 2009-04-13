#pragma once
#include "nodes.h"

namespace ntree
{

struct RayTracer
{
  int isoLevel;
  const Node * root;

  // result
  bool hit;
  cg::point_3f pos;
  cg::point_3f normal;
  cg::point_3f color;

  void Trace(point_3f p, point_3f dir)
  {
    hit = false;
    
    

  }


};





}