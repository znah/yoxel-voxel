#pragma once

#include "gpu_nodes.h"

namespace ntree
{

typedef uchar4 ValueType;
const ValueType DefValue = {0, 0, 0, 0};


struct Node;
typedef Node * NodePtr;
struct Node
{
  NodePtr parent;
  ValueType * data;
  NodePtr   * child;

  // gpu data
  GPURef gpuData;
  GPURef gpuChild;
};


}