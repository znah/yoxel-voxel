#pragma once

#include "nodes.h"

namespace ntree
{

////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// tree utils ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


struct RangeBuilder
{
  range_3i dstRange;
  const ValueType * data;

  void build(Node & node, int voxSize, point_3i posOnLevel)
  {
    point_3i voxPos = posOnLevel * voxSize;
    range_3i range(voxPos, voxSize+BrickBoundary);
    if (!range.intersects(dstRange))
      return;

    Assert(voxSize >= BrickSize-BrickBoundary);
    if (voxSize == BrickSize-BrickBoundary)
    {
      node.MakeBrick();
      updateBrick(node.brickPtr(), range);
    }
    else
    {
      node.MakeGrid();
      point_3i p = GridSize * posOnLevel;
      int sz = voxSize / 2;
      for (walk_3 i(GridSize); !i.done(); ++i)
        build(node.child(i.p), sz, p + i.p);
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