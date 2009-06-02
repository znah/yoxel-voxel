#pragma once

const uint32 ZeroBlock    = 0xFFFFFFF0;
const uint32 FullBlock    = 0xFFFFFFF1;
const uint32 BrickRefMask = 0x80000000;

struct BitTree
{
  struct Brick { uint32 lo, hi; };

  uint32 root;
  std::vector<uint32> grids;
  std::vector<Brick> bricks;
};

struct TraceResult
{
  bool hit;
  point_3i pos;
};


struct RayTracer
{
  const BitTree & tree;
  RayTracer(const BitTree & atree) : tree(atree) {}

  point_3f dirSign;


  TraceResult trace(point_3f start, point_3f dir)
  {
    

  }
};
