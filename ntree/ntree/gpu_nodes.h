#pragma once

typedef uint GPURef;
const GPURef GPUNull = 0xffffffff;

namespace ntree
{

const int NodeSizePow = 2;
const int NodeSize = 1 << NodeSizePow;
const int NodeSize3 = NodeSize*NodeSize*NodeSize;

}