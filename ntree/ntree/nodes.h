#pragma once

namespace ntree
{

const int NodeSizePow = 2;
const int NodeSize = 1 << NodeSizePow;
const int NodeSize3 = NodeSize*NodeSize*NodeSize;
typedef uchar4 ValueType;
const ValueType DefValue = {0, 0, 0, 0};

const uint NullGpuRef = 0xffffffff;

struct Node;
typedef Node * NodePtr;
struct Node
{
  NodePtr parent;
  ValueType * data;
  NodePtr   * child;

  // gpu data
  uchar4 gpuData;
  uchar4 gpuChild;
};


}