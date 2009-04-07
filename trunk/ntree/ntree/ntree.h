#pragma once

#include "nodes.h"

namespace ntree
{

////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// tree utils ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

inline void PrepareData(point_3i size, int isoLevel, ValueType * data)
{
  int count = size.x * size.y * size.z;
  std::vector<uchar> mark(count, 0);

  int tLo(255), tHi(0);
  for ( walk_3 i(size - point_3i(1, 1, 1)); !i.done(); ++i )
  {
    int ofs = (size.y * i.z() + i.y()) * size.x + i.x();
    int lo(255), hi(0);
    for (int i = 0; i < 8; ++i)
    {
      int v = data[ofs + (i&1) + ((i>>1)&1)*size.x + ((i>>2)&1)*size.y].w;
      lo = cg::min(v, lo);
      hi = cg::max(v, hi);
    }
    if (lo > isoLevel || hi < isoLevel)
      continue;
    mark[ofs] = 1;
    tLo = cg::min(tLo, lo);
    tHi = cg::max(tHi, hi);
  }
  int dv = tHi - tLo;
  Assert(dv > 0);
  for (int i = 0; i < count; ++i)
  {
    int v = data[i].w;
    v = (v - tLo) * 255 / dv;
    v = cg::bound(v, 0, 255);
    v = (v & 0xfe) + mark[i];
    data[i].w = v;
  }
}


struct RangeBuilder
{
  range_3i dstRange;
  const ValueType * data;

  void build(Node & node, int sceneVoxSize)
  {
    buildNode(node, sceneVoxSize, point_3i(0, 0, 0));
  }


  void buildNode(Node & node, int nodeVoxSize, point_3i posOnLevel)
  {
    point_3i voxPos = posOnLevel * nodeVoxSize;
    range_3i range(voxPos, nodeVoxSize+BrickBoundary);
    if (!range.intersects(dstRange))
      return;

    Assert(nodeVoxSize >= BrickSize-BrickBoundary);
    if (nodeVoxSize == BrickSize-BrickBoundary)
    {
      node.MakeBrick();
      updateBrick(node.brickPtr(), range);
    }
    else
    {
      node.MakeGrid();
      point_3i p = GridSize * posOnLevel;
      int sz = nodeVoxSize / 2;
      for (walk_3 i(GridSize); !i.done(); ++i)
        buildNode(node.child(i.p), sz, p + i.p);
    }
    node.Shrink(false);
  }

  void updateBrick(ValueType * dstBuf, const range_3i & nodeRange)
  {
    range_3i updateRange = nodeRange;
    updateRange &= dstRange;
    point_3i upd2node = updateRange.p1 - nodeRange.p1;
    point_3i node2data = nodeRange.p1 - dstRange.p1;
    point_3i srcSize = dstRange.size();
    for (walk_3 i(updateRange.size()); !i.done(); ++i)
    {
      point_3i dst = i.p + upd2node;
      point_3i src = dst + node2data;

      int srcOfs = (src.z * srcSize.y + src.y) * srcSize.x + src.x;
      int dstOfs = (dst.z * BrickSize + dst.y) * BrickSize + dst.x;
      dstBuf[dstOfs] = data[srcOfs];
    }
  }
};

struct StatsBuilder
{
  int grids, bricks;

  StatsBuilder() : grids(0), bricks(0) {}

  void walk(Node & node)
  {
    if (node.GetType() == Node::Brick)
      ++bricks;
    if (node.GetType() == Node::Grid)
    {
      ++grids;
      for (int i = 0; i < GridSize3; ++i)
        walk(node.child(i));
    }
  }
};

}