#pragma once

typedef uint GPURef;
const GPURef GPUNull = 0xffffffff;


namespace ntree
{

const int NodeSizePow = 2;
const int NodeSize = 1 << NodeSizePow;
const int NodeSize3 = NodeSize*NodeSize*NodeSize;
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