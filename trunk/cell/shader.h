#pragma once

#include "vox_node.h"

struct SimpleShader
{
  point_3f lightPos;
  point_3f viewerPos;

  Color32 Shade(VoxData data, const point_3f & dir, float t) const
  {
    Normal16 n16;
    Color16 c16;
    UnpackVoxData(data, c16, n16);
    point_3f n = UnpackNormal(n16);
    Color32 c = UnpackColor(c16);
    float diff = max(0.0f, -n*dir);
    return c*diff;
  };
};