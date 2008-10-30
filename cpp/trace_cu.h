#pragma once

#include "vox_node.h"

struct VoxStructTree
{
  VoxNodeId     root;
  VoxNode     * nodes;
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
  int endNodeChild;
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


