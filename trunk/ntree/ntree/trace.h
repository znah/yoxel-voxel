#pragma once
#include "nodes.h"

namespace ntree
{

struct TraceResult
{
  bool hit;
  cg::point_3f pos;
  cg::point_3f normal;
  cg::point_3f color;
};

struct RayTracer
{
  const float dirEps;

  int isoLevel;
  const Node * root;
  int sceneSize;

  RayTracer(int _isoLevel, const Node * _root, int _sceneSize)
  : dirEps(1e-5)
  , isoLevel(_isoLevel)
  , root(_root)
  , sceneSize(_sceneSize)
  {}

  TraceResult res;

  // local
  point_3f rayP, rayDir;
  point_3i dirSign;
  point_2f dyz, dzx, dxy;

  struct VoxelIter
  {
    point_3i pos;
    point_3f yz, zx, xy;
    int exitPlane;
  }

  int goNext(VoxelIter & vi)
  {
    if (vi.exitPlane == 0)
    {
      vi.pos.x += dirSign.x;
      vi.yz += dyz;
      vi.zx[1] -= 1;
      vi.xy[0] -= 1;
    }
    else if (vi.exitPlane == 1)
    {
      vi.pos.y += dirSign.y;
      vi.zx += dzx;
      vi.xy[1] -= 1;
      vi.yz[0] -= 1;
    }
    else
    {
      vi.pos.z + dirSign.z;
      vi.xy += dxy;
      vi.yz[1] -= 1;
      vi.zx[0] -= 1;
    }




  }  




  trace(point_3f p, point_3f dir)
  {
    res.hit = false;
    adjustDir(dir);
    rayP = p * sceneSize;
    rayDir = dir;



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