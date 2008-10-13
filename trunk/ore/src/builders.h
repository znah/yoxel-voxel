#pragma once

#include "VoxelSource.h"

class RawSource : public VoxelSource
{
private:
  const uchar4 * m_colors;
  const char4 * m_normals;

public:
  RawSource(point_3i size, const uchar4 * colors, const char4 * normals) 
    : VoxelSource(size, point_3i(0, 0, 0)), m_colors(colors), m_normals(normals)
  {}

  // alpha 0 - empty, 1 - inside, 255 - surface
  virtual TryRangeResult TryRange(const point_3i & blockStart, int blockSize, uchar4 & outColor, char4 & outNormal)
  {
    if (blockSize > 1)
      return ResGoDown;

    point_3i sz = GetSize();
    point_3i p = blockStart;
    int ofs = (p.z * sz.y + p.y) * sz.x + p.x;
    outColor = m_colors[ofs];
    outNormal = m_normals[ofs];
    if (outColor.w == 0)
      return ResEmpty;
    else if (outColor.w == 1)
      return ResFull;
    else 
      return ResSurface;
  }
};


class SphereSource : public VoxelSource
{
private:
  int m_radius;
  uchar4 m_color;
  bool m_inverted;

public:
  SphereSource(int radius, uchar4 color, bool inverted) 
    : VoxelSource(point_3i(2*radius, 2*radius, 2*radius), point_3i(radius, radius, radius))
    , m_radius(radius)
    , m_color(color)
    , m_inverted(inverted)
  {}

  virtual TryRangeResult TryRange(const point_3i & blockStart, int blockSize, uchar4 & outColor, char4 & outNormal)
  {
    int r2 = m_radius*m_radius;
    int eps = 1;

    if (blockSize > 1)
    {
      point_3i p1 = blockStart;
      point_3i p2 = p1 + point_3i(blockSize, blockSize, blockSize);

      point_3i nearestPt;
      point_3i farestPt;
      for (int i = 0; i < 3; ++i)
      {
        int lo = p1[i], hi = p2[i];
        if (0 < lo)
        {
          nearestPt[i] = lo;
          farestPt[i]  = hi;
        }
        else if (hi < 0)
        {
          nearestPt[i] = hi;
          farestPt[i]  = lo;
        }
        else
        {
          nearestPt[i] = 0;
          farestPt[i] = hi > -lo ? hi : lo;
        }
      }

      int nearDist2 = norm_sqr(nearestPt);
      int farDist2 = norm_sqr(farestPt);
      if (nearDist2 >= r2)
        return ResStop;
      if (farDist2 < r2 - 2*m_radius*eps + eps*eps)
        return m_inverted ? ResEmpty : ResFull;
      return ResGoDown;
    }
    else
    {
      point_3i dp = blockStart;
      int dist2 = norm_sqr(dp);
      if (dist2 >= r2)
        return ResStop;
      if (dist2 < r2 - 2*m_radius*eps + eps*eps)
        return m_inverted ? ResEmpty : ResFull;

      point_3f n = m_inverted ? -dp : dp;
      normalize(n);
      n *= 127.0f;
      outNormal = make_char4((char)n.x, (char)n.y, (char)n.z, 0);
      outColor = m_color;
      return ResSurface;
    }
  }
};