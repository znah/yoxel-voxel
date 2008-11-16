#pragma once

#include "vox_node.h"

struct VoxStructTree
{
  VoxNodeId     root;
  VoxNode     * nodes;
};

struct RenderParams
{
  float detailCoef;

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
