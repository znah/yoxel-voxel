#pragma once

#include <cuda_runtime.h>

#pragma pack(push, 4)

typedef unsigned int uint;
typedef unsigned char uchar;

typedef int VoxNodeId;

const VoxNodeId EmptyNode = -1;
const VoxNodeId FullNode  = -2;

union VoxNodeInfo
{
  uint pad;
  struct
  {
  uchar selfChildId:3;
  bool  emptyFlag:1;
  };
};

#define VOX_LEAF 0x40000000
#define IDX(id) ((id) & 0x3fffffff)

struct VoxNode
{
  VoxNodeInfo flags;
  VoxNodeId   parent;
  uchar4      color;
  char4       normal;
  VoxNodeId   child[8];
};

struct VoxLeaf
{
  uchar4      color;
  char4       normal;
};


struct VoxStructTree
{
  VoxNodeId     root;

  VoxNode     * nodes;
  VoxLeaf     * leafs;
};

struct RenderParams
{
  float3 eye;
  float3 dir;
  float3 right;
  float3 up;

  float3 lightPos;
};

struct RayData
{
  float3 dir;
  float t;
  VoxNodeId endNode;
  float endNodeSize;
};

struct TraceParams
{
  float3 start;
  
  float detailCoef;
  VoxNodeId startNode;
  float3 startNodePos;
  float  startNodeSize;
  int startNodeLevel;
};

#pragma pack(pop)

