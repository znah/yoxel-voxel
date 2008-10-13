#pragma once

#include "point.h"

enum TryRangeResult { ResStop, ResGoDown, ResEmpty, ResFull, ResSurface };

class VoxelSource
{
private:
  point_3i m_size;
  point_3i m_pivot;
public:
  VoxelSource(point_3i size, point_3i pivot) : m_size(size), m_pivot(pivot) {}
  point_3i GetSize() const { return m_size; }
  point_3i GetPivot() const { return m_pivot; }

  virtual TryRangeResult TryRange(const point_3i & blockStart, int blockSize, uchar4 & outColor, char4 & outNormal)
  {
    return ResStop;
  }
};
