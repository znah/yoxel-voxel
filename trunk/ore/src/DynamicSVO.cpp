#include "DynamicSVO.h"

#include <fstream>
#include <iostream>

#include "common/grid_walk.h"
#include "utils.h"

using namespace cg;
using std::cout;
using std::endl;


void DynamicSVO::Save(std::string filename)
{
  std::ofstream file(filename.c_str(), std::ios::binary);
  m_nodes.save(file);
  m_leafs.save(file);
  write(file, m_root);
}

bool DynamicSVO::Load(std::string filename)
{
  std::ifstream file(filename.c_str(), std::ios::binary);
  m_nodes.load(file);
  m_leafs.load(file);
  read(file, m_root);

  m_curVersion = 0;
  return true;
}


void DynamicSVO::DelNode(VoxNodeId node, bool recursive)
{
  if (IsNull(node))
    return;
  if (IsLeaf(node))
    m_leafs.erase(ToIdx(node));
  else
  {
    if (recursive)
    {
      VoxNodeId children[8];
      FetchChildren(node, children);
      for (int i = 0; i < 8; ++i)
        DelNode(children[i]);
    }
    m_nodes.erase(ToIdx(node));
  }
}

VoxNodeId DynamicSVO::SetLeaf(VoxNodeId node, uchar4 color, char4 normal)
{
  if (!IsLeaf(node))
  {
    DelNode(node);
    node = m_leafs.insert() | VOX_LEAF;
  }
  VoxLeaf & leaf = m_leafs[ToIdx(node)];
  m_leafs.setItemVer(ToIdx(node), m_curVersion);
  leaf.color = color;
  leaf.normal = normal;
  return node;
}

void DynamicSVO::FetchChildren(VoxNodeId node, VoxNodeId * dst) const
{
  if (IsNull(node))
    std::fill(dst, dst+8, node);
  else if (IsLeaf(node))
    std::fill(dst, dst+8, FullNode);
  else
  {
    const VoxNode & nd = m_nodes[ToIdx(node)];
    std::copy(nd.child, nd.child+8, dst);
  }
}

VoxNodeId DynamicSVO::UpdateChildren(VoxNodeId node, const VoxNodeId * children)
{
  if (std::count_if(children, children+8, IsNull) == 8)
  {
    DelNode(node, false);
    return children[0]; // slight cheat;
  }

  if (!IsNode(node))
  {
    DelNode(node, false);
    node = m_nodes.insert();
    VoxNode & nd = m_nodes[ToIdx(node)];
    m_nodes.setItemVer(ToIdx(node), m_curVersion);
    nd.parent = -1;
    nd.flags.pad = 0;
  }

  int count = 0;
  point_3i accCol;
  point_3f accNorm;
  for (int i = 0; i != 8; ++i)
  {
    VoxNodeId ch = children[i];
    if (IsNull(ch))
      continue;

    uchar4 c;
    char4 n;

    if (IsLeaf(ch))
    {
      VoxLeaf & leaf = m_leafs[ToIdx(ch)];
      m_leafs.setItemVer(ToIdx(ch), m_curVersion);
      c = leaf.color;
      n = leaf.normal;
    }
    else
    {
      VoxNode & nd = m_nodes[ToIdx(ch)];
      m_nodes.setItemVer(ToIdx(ch), m_curVersion);
      nd.parent = node;
      nd.flags.selfChildId = i;
      c = nd.color;
      n = nd.normal;
    }
    
    accCol  += point_3i(c.x, c.y, c.z);
    accNorm += point_3f(n.x, n.y, n.z);
    ++count;
  }

  accCol /= count;
  accNorm /= (float)count;
  normalize(accNorm);
  accNorm *= 127.0f;

  VoxNode & nd = m_nodes[ToIdx(node)];
  m_nodes.setItemVer(ToIdx(node), m_curVersion);
  nd.color = make_uchar4(accCol.x, accCol.y, accCol.z, count >= 1 ? 255 : 0);
  nd.normal = make_char4((int)accNorm.x, (int)accNorm.y, (int)accNorm.z, 0);
  nd.flags.emptyFlag = false;
  std::copy(children, children+8, nd.child);

  return node;
}


template <class RangeSampler>
struct DynamicSVO::TreeBuilder
{
  DynamicSVO & svo;
  RangeSampler  & sampler;
  BuildMode mode;
  int destLevel;

  TreeBuilder(DynamicSVO * svo_, RangeSampler & sampler_, BuildMode mode_) 
    : svo(*svo_) 
    , sampler(sampler_)
    , mode(mode_)
  {}

  VoxNodeId BuildRange(int level, const point_3i & p, VoxNodeId node)
  {
    if (mode == BUILD_MODE_GROW && node == FullNode)
      return node;
    if (mode == BUILD_MODE_CLEAR && node == EmptyNode)
      return node;

    uchar4 col; 
    char4 n;
    TryRangeResult res = sampler.TryRange(level, p, col, n);
    if (res == ResEmpty && mode == BUILD_MODE_CLEAR)
    {
      svo.DelNode(node);
      node = EmptyNode;
    }

    if (res == ResFull && mode == BUILD_MODE_GROW)
    {
      svo.DelNode(node);
      node = FullNode;
    }

    if (res == ResSurface)
    {
      if (!IsLeaf(node) || mode == BUILD_MODE_GROW)
        node = svo.SetLeaf(node, col, n);
    }

    if (res == ResGoDown)
    {
      VoxNodeId children[8];
      svo.FetchChildren(node, children);
      for (walk_3 i(2, 2, 2); !i.done(); ++i)
        children[i.flat()] = BuildRange(level+1, p*2+i.p, children[i.flat()]);
      node = svo.UpdateChildren(node, children);
    }
    return node;
  }
};

struct RawSampler
{
  int destLevel;
  point_3i p1, p2, size;
  const uchar4 * colors;
  const char4 * normals;

  TryRangeResult TryRange(int level, const point_3i & p, uchar4 & outColor, char4 & outNormal)
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

void DynamicSVO::BuildRange(int level, const cg::point_3i & origin, const cg::point_3i & size, const uchar4 * colors, const char4 * normals, BuildMode mode)
{
  ++m_curVersion;

  RawSampler sampler;
  sampler.destLevel = level;
  sampler.p1 = origin;
  sampler.p2 = origin + size;
  sampler.size = size;
  sampler.colors = colors;
  sampler.normals = normals;

  TreeBuilder<RawSampler> bld(this, sampler, mode);
  m_root = bld.BuildRange(0, point_3i(0, 0, 0), m_root);
}


inline float max(const point_3f & p) { return max(p.x, max(p.y, p.z)); }
inline float min(const point_3f & p) { return min(p.x, min(p.y, p.z)); }

inline int argmin(const point_3f & p) 
{
  if (p.x > p.y)
    return (p.y < p.z) ? 1 : 2;
  else
    return (p.x < p.z) ? 0 : 2;
}

inline int argmax(const point_3f & p) 
{
  if (p.x < p.y)
    return (p.y > p.z) ? 1 : 2;
  else
    return (p.x > p.z) ? 0 : 2;
}

template<class T>
inline void swap(T & a, T & b) { T c = a; a = b; b = c; }

VoxNodeId DynamicSVO::RecTrace(VoxNodeId node, point_3f t1, point_3f t2, const uint dirFlags, float & t) const
{
  if (IsNull(node))
    return EmptyNode;
  if (min(t2) <= 0)
    return EmptyNode;
  float tEnter = max(t1);
  if (IsLeaf(node) || node == FullNode)
  {
    t = tEnter;
    return node;
  }
  
  point_3f tm = 0.5f*(t1+t2);
  int ch = 0;
  for (int i = 0; i < 3; ++i)
  {
    if (tEnter < tm[i])
      t2[i] = tm[i];
    else
    {
      t1[i] = tm[i];
      ch |= 1<< i;
    }
  }
  
  VoxNodeId children[8];
  FetchChildren(node, children);

  while (true)
  {
    VoxNodeId res = RecTrace(children[ch^dirFlags], t1, t2, dirFlags, t);
    if (!IsNull(res))
      return res;
    int exitPlane = argmin(t2);
    if (ch & (1<<exitPlane))
      return EmptyNode;
    ch |= 1<<exitPlane;
    float dt = t2[exitPlane] - t1[exitPlane];
    t1[exitPlane] = t2[exitPlane];
    t2[exitPlane] += dt;
  }
}

float DynamicSVO::TraceRay(const point_3f & p, point_3f dir) const
{
  const float eps = 1e-8f;
  for (int i = 0; i < 3; ++i)
    if (abs(dir[i]) < eps)
      dir[i] = (dir[i] < 0) ? -eps : eps;

  point_3f t1 = (point_3f(0, 0, 0) - p) / dir;
  point_3f t2 = (point_3f(1, 1, 1) - p) / dir;
  uint dirFlags = 0;
  for (int i = 0; i < 3; ++i)
    if (dir[i] < 0)
    {
      dirFlags |= 1<<i;
      swap(t1[i], t2[i]);
    }

  float t = 0;
  if (max(t1) >= min(t2))
    return t;
  RecTrace(m_root, t1, t2, dirFlags, t);
  
  return t;
}


struct SphereSampler
{
  int destLevel;
  int radius;
  point_3i pos;
  uchar4 color;
  bool inverted;


  TryRangeResult TryRange(int level, const point_3i & p, uchar4 & outColor, char4 & outNormal)
  {
    int r2 = radius*radius;
    int eps = 1;

    if (level < destLevel)
    {
      point_3i p1 = p * (1<<(destLevel - level));
      int sd = (1 << (destLevel - level)) - 1;
      point_3i p2 = p1 + point_3i(sd, sd, sd);

      point_3i nearestPt;
      point_3i farestPt;
      for (int i = 0; i < 3; ++i)
      {
        int x = pos[i], lo = p1[i], hi = p2[i];
        if (x < lo)
        {
          nearestPt[i] = lo;
          farestPt[i]  = hi;
        }
        else if (hi < x)
        {
          nearestPt[i] = hi;
          farestPt[i]  = lo;
        }
        else
        {
          nearestPt[i] = x;
          farestPt[i] = x-lo < hi-x ? hi : lo;
        }
      }

      int nearDist2 = norm_sqr(pos - nearestPt);
      int farDist2 = norm_sqr(pos - farestPt);
      if (nearDist2 >= r2)
        return ResStop;
      if (farDist2 < r2 - 2*radius*eps + eps*eps)
        return inverted ? ResEmpty : ResFull;
      return ResGoDown;
    }
    else
    {
      point_3i dp = p - pos;
      int dist2 = norm_sqr(dp);
      if (dist2 >= r2)
        return ResStop;
      if (dist2 < r2 - 2*radius*eps + eps*eps)
        return inverted ? ResEmpty : ResFull;

      point_3f n = inverted ? -dp : dp;
      normalize(n);
      n *= 127.0f;
      outNormal = make_char4((char)n.x, (char)n.y, (char)n.z, 0);
      outColor = color;
      return ResSurface;
    }
  }
};

void DynamicSVO::BuildSphere(int level, int radius, const cg::point_3i & pos, BuildMode mode)
{
  ++m_curVersion;

  SphereSampler sampler;
  sampler.destLevel = level;
  sampler.radius = radius;
  sampler.pos = pos;
  sampler.color = make_uchar4((mode == BUILD_MODE_CLEAR) ? 255 : 0, 128, 128, 255);
  sampler.inverted = (mode == BUILD_MODE_CLEAR);

  TreeBuilder<SphereSampler> bld(this, sampler, mode);
  m_root = bld.BuildRange(0, point_3i(0, 0, 0), m_root);
}

int DynamicSVO::CountChangedPages() const
{
  return m_leafs.countPages(m_curVersion) + m_nodes.countPages(m_curVersion);
}

int DynamicSVO::CountTransfrerSize() const
{
  int count = m_leafs.countPages(m_curVersion) * m_leafs.getPageSize();
  count += m_nodes.countPages(m_curVersion) * m_nodes.getPageSize();
  return count;
}
