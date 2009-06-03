#pragma once

const uint32 ZeroBlock    = 0xFFFFFFF0;
const uint32 FullBlock    = 0xFFFFFFF1;
const uint32 BrickRefMask = 0x80000000;

const int BrickSize = 4;
const int GridSize  = 4;
const int GridSize3 = GridSize * GridSize * GridSize;

struct Brick { uint32 lo, hi; };
typedef uint32 Grid[GridSize3];

inline float maxCoord(const point_3f & p) { return cg::max(p.x, cg::max(p.y, p.z)); }
inline float minCoord(const point_3f & p) { return cg::min(p.x, cg::min(p.y, p.z)); }


struct RayTracer
{
  const Brick * m_bricks;
  const Grid * m_grids;
  const int m_root;

  RayTracer(const Brick * bricks, const Grid * grids, int root) 
  : m_bricks(bricks), m_grids(grids), m_root(root) {}

  // result
  bool hit;
  point_3i voxelPos;
  
  // ray data
  point_3f invDir;
  point_3f eye;


  void trace(point_3f eye_, point_3f dir)
  {
    hit = false;
    voxelPos = point_3i(0, 0, 0);

    adjustDir(dir);
    invDir = point_3f(1.0f, 1.0f, 1.0f) / dir;
    eye = eye_;

    traceGrid(m_root, point_3i(0, 0, 0), 1.0f);
  }

  void traceGrid(uint32 gridId, point_3i gridPos, float gridWldSize)
  {
    point_3f lo = gridPos * gridWldSize; 
    point_3f hi = lo + point_3f(gridWldSize, gridWldSize, gridWldSize);
    
    point_3f t1 = (lo - eye) & invDir;
    point_3f t2 = (hi - eye) & invDir;
    for (int i = 0; i < 3; ++i)
      cg::sort2(t1[i], t2[i]);

    float tenter = maxCoord(t1);
    float texit  = minCoord(t2);

    if (tenter > texit || texit < 0)
      return;
    hit = true;

  }


  void adjustDir(point_3f & dir)
  {
    const float eps = 1e-8f;
    for (int i = 0; i < 3; ++i)
      if (fabs(dir[i]) < eps)
        dir[i] = (dir[i] < 0) ? -eps : eps;
  }
};
