#pragma once

const uint32 ZeroBlock    = 0xFFFFFFF0;
const uint32 FullBlock    = 0xFFFFFFF1;
const uint32 BrickRefMask = 0x80000000;

const int BrickSize = 4;
const int GridSize  = 4;
const int GridSize3 = GridSize * GridSize * GridSize;

struct Brick { uint32 lo, hi; };
struct Grid { uint32 child[GridSize3]; };

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

    hit = traceNode(m_root, point_3i(0, 0, 0), 1.0f);
    if (hit)
      printf("hit\n");
  }

  bool traceNode(uint32 nodeId, point_3i nodePos, float nodeWldSize)
  {
    if (nodeId == ZeroBlock)
      return false;
    point_3f lo = nodePos * nodeWldSize; 
    point_3f hi = lo + point_3f(nodeWldSize, nodeWldSize, nodeWldSize);
    
    point_3f t1 = (lo - eye) & invDir;
    point_3f t2 = (hi - eye) & invDir;
    for (int i = 0; i < 3; ++i)
      cg::sort2(t1[i], t2[i]);

    float tenter = maxCoord(t1);
    float texit  = minCoord(t2);

    if (tenter > texit || texit < 0)
      return false;
    if ((nodeId & BrickRefMask) != 0)
      return true;

    printf("%d, %d, %d  %f\n", nodePos.x, nodePos.y, nodePos.z, nodeWldSize);
    //if (nodeWldSize < 0.3)
    //  return true;

    point_3i subNodePos = nodePos * GridSize;
    float subNodeSize = nodeWldSize / GridSize;
    for (int z = 0; z < GridSize; ++z)
    for (int y = 0; y < GridSize; ++y)
    for (int x = 0; x < GridSize; ++x)
    {
      int nx = (invDir.x >= 0) ? x : GridSize - x;
      int ny = (invDir.y >= 0) ? y : GridSize - y;
      int nz = (invDir.z >= 0) ? z : GridSize - z;
      int ch = (nz * GridSize + ny) * GridSize + nx;
      if ( traceNode(m_grids[nodeId].child[ch], subNodePos + point_3i(nx, ny, nz), subNodeSize) )
        return true;
    }
    return false;
  }


  void adjustDir(point_3f & dir)
  {
    const float eps = 1e-8f;
    for (int i = 0; i < 3; ++i)
      if (fabs(dir[i]) < eps)
        dir[i] = (dir[i] < 0) ? -eps : eps;
  }
};
