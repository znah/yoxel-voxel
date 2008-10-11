#pragma once

#include "DynamicSVO.h"

#include "range.h"

class RawBuilder : public VoxelSource
{
private:
  range_3i m_range;
  const uchar4 * colors;
  const char4 * normals;

public:

  virtual TryRangeResult TryRange(const point_3i & blockPos, int blockSize, uchar4 & outColor, char4 & outNormal)
  {



    point_3i pd = p * (1<<(destLevel - level));
    int sd = 1 << (destLevel - level);
    if (pd.x >= p2.x || pd.x+sd <= p1.x) return ResStop;
    if (pd.y >= p2.y || pd.y+sd <= p1.y) return ResStop;
    if (pd.z >= p2.z || pd.z+sd <= p1.z) return ResStop;

    if (level < destLevel)
      return ResGoDown;

    point_3i dp = p - p1;
    int ofs = (dp.z*size.y + dp.y)*size.x + dp.x;
    outColor = colors[ofs];
    outNormal = normals[ofs];

    if (outColor.w == 0)
      return ResEmpty;
    if (outColor.w == 1)
      return ResFull;
    return ResSurface;
  }
};
