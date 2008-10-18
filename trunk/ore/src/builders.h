#pragma once

#include "VoxelSource.h"
#include "range.h"

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

class IsoSource : public VoxelSource
{
private:
  const uchar * m_data;
  int m_isolevel;
  bool m_lowInside;
  uchar4 m_color;

  uchar trans(uchar v) const { return m_lowInside ? 255-v : v; }

  int ofs(const point_3i & p) const
  {
    point_3i sz = GetSize();
    return (p.z * sz.y + p.y) * sz.x + p.x;
  }

  uchar get(point_3i p) const
  {
    point_3i sz = GetSize();
    for (int i = 0; i < 3; ++i)
      p[i] = cg::bound(p[i], 0, sz[i]-1);
    return trans(m_data[ofs(p)]);
  }

  void getrange(range_3i rng, int & lo, int & hi)
  {
    lo = 255;
    hi = 0;
    rng &= range_3i(point_3i(0, 0, 0), GetSize());
    if (rng.empty())
      return;

    for (int z = rng.p1.z; z < rng.p2.z; ++z)
      for (int y = rng.p1.y; y < rng.p2.y; ++y)
      {
        int p = ofs(point_3i(0, y, z));
        for (int x = rng.p1.x; x < rng.p2.x; ++x)
        {
          int v = trans(m_data[p+x]);
          lo = cg::min(lo, v);
          hi = cg::max(hi, v);
        }
      }
  }

public:
  IsoSource(point_3i size, const uchar * data)
    : VoxelSource(size, point_3i(0, 0, 0))
    , m_data(data)
    , m_isolevel(128)
    , m_lowInside(false)
    , m_color(make_uchar4(128, 128, 128, 255))
  {}

  void SetIsoLevel(int isolevel) { m_isolevel = isolevel; }
  void SetInside(bool low) { m_lowInside = low; }
  void SetColor(uchar4 color) { m_color = color; }

  virtual TryRangeResult TryRange(const point_3i & blockStart, int blockSize, uchar4 & outColor, char4 & outNormal)
  {
    int level = trans(m_isolevel);

    if (blockSize > 1)
    {
      range_3i rng(blockStart - point_3i(1, 1, 1), blockSize+2);
      int lo, hi;
      getrange(rng, lo, hi);

      if (hi < level)
        return ResEmpty;
      if (level <= lo)
        return ResFull;
      return ResGoDown;
    }
    else
    {
      const point_3i & p = blockStart;
      int v = get(p);
      if (v < level)
        return ResEmpty;

      point_3f n;
      bool inside = true;
      for (int i = 0; i < 3; ++i)
      {
        point_3i dp(0, 0, 0);
        dp[i] = 1;
        int v1 = get(p - dp);
        int v2 = get(p + dp);
        n[i] = -0.5f*(v2-v1);
        inside = inside && (v1 >= level) && (v2 >= level);
      }

      if (inside)
        return ResFull;

      outColor = m_color;
      normalize(n);
      n *= 127.0f;
      outNormal = make_char4((char)n.x, (char)n.y, (char)n.z, 0);
      return ResSurface;
    }
  }
};
