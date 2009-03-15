#pragma once

#include "nodes.h"

namespace ntree
{

////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// tree utils ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

NodePtr DelTree(NodePtr node)
{
  if (node == NULL)
    return NULL;
  if (node->child != NULL)
  {
    NodePtr * pp = node->child;
    for (uint ci = 0; ci < NodeSize3; ++ci, ++pp)
      (*pp) = DelTree(*pp);
  }
  delete [] node->child;
  delete [] node->data;
  delete node;
  return NULL;
}

inline NodePtr createNode(bool hasChildren)
{
  NodePtr node = new Node;
  node->data = new ValueType[NodeSize3];
  std::fill(node->data, node->data + NodeSize3, DefValue);
  if (hasChildren)
  {
    node->child = new NodePtr[NodeSize3];
    std::fill(node->child, node->child + NodeSize3, (NodePtr)NULL);
  }
  else
    node->child = NULL;
  
  return node;
}

inline point_3i ci2pt(uint ci)
{
  const uint mask = NodeSize-1;
  return point_3i(ci & mask, (ci>>NodeSizePow) & mask, (ci>>(2*NodeSizePow)) & mask);
}

inline uint pt2ci(const point_3i & pt)
{
  uint ci = pt.x | (pt.y << NodeSizePow) | (pt.z << (2*NodeSizePow));
  return ci;
}


struct RangeBuilder
{
  range_3i dstRange;
  const ValueType * data;

  NodePtr build(NodePtr root, int depth)
  {
    ValueType t;
    return updateNode(root, depth-1, 1<<(NodeSizePow*depth), point_3i(0, 0, 0), t);
  }

  NodePtr updateNode(NodePtr node, int level, int voxSize, const point_3i & voxPos, ValueType & nodeVal)
  {
    range_3i range(voxPos, voxSize);
    if (!range.intersects(dstRange))
      return node;

    if (level == 0)
      return updateLeaf(node, range, nodeVal);
    else
      return updateGrid(node, level, voxSize, voxPos, nodeVal);
  }

  NodePtr updateGrid(NodePtr node, int level, int voxSize, const point_3i & voxPos, ValueType & nodeVal)
  {
    if (node == NULL)
      node = createNode(true);

    Assert(node->child != NULL);
    int chSize = voxSize / NodeSize;

    NodePtr * pp = node->child;
    bool allNull = true;
    for (uint ci = 0; ci < NodeSize3; ++ci, ++pp)
    {
      (*pp) = updateNode(*pp, level-1, chSize, voxPos + ci2pt(ci) * chSize, node->data[ci]);
      allNull &= ((*pp) == NULL);
    }

    if (!calcNodeVal(node->data, nodeVal) && allNull)
    {
      delete node;
      return NULL;
    }
    //static int count = 0;
    //printf("node: %d %d %d; %d - %d\n", voxPos.x, voxPos.y, voxPos.z, level, ++count);
    return node;
  }

  NodePtr updateLeaf(NodePtr node, const range_3i & range, ValueType & nodeVal)
  {
    if (node == NULL)
      node = createNode(false);

    Assert(node->child == NULL);

    updateLeafData(node->data, range);
    if (!calcNodeVal(node->data, nodeVal))
    {
      delete node;
      return NULL;
    }
    //static int count = 0;
    //printf("leaf: %d %d %d - %d\n", range.p1.x, range.p1.y, range.p1.z, ++count);
    return node;
  }

  void updateLeafData(ValueType * dstBuf, const range_3i & nodeRange)
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

      int dstOfs = (dst.z * NodeSize  + dst.y) * NodeSize  + dst.x;
      int srcOfs = (src.z * srcSize.y + src.y) * srcSize.x + src.x;
      dstBuf[dstOfs] = data[srcOfs];
    }
  }

  bool calcNodeVal(ValueType * data, ValueType & nodeVal)
  {
    point_4i accum;
    bool allEqual = true;
    point_4i first = point_4i(data->x, data->y, data->z, data->w);
    for (ValueType * p = data; p != data+NodeSize3; ++p)
    {
      point_4i v = point_4i(p->x, p->y, p->z, p->w);
      accum += v;
      allEqual &= (first == v);
    }
    accum /= NodeSize3;
    nodeVal.x = accum.x;
    nodeVal.y = accum.y;
    nodeVal.z = accum.z;
    nodeVal.w = accum.w;
    return !allEqual;
  }
};

struct StatsBuilder
{
  std::vector<int> count;

  void walk(NodePtr node, int level)
  {
    if (node == NULL)
      return;

    if ((int)count.size() < level+1)
      count.resize(level+1, 0);
    ++count[level];
    
    if (node->child != NULL)
    {
      NodePtr * pp = node->child;
      for (uint ci = 0; ci < NodeSize3; ++ci, ++pp)
        walk(*pp, level+1);
    }
  }
};

}