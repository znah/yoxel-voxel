#include "DynamicSVO.h"

#include <fstream>
#include <iostream>

#include "common/grid_walk.h"
#include "utils.h"
#include "range.h"

using namespace cg;
using std::cout;
using std::endl;

inline int ToIdx(VoxNodeId node) { Assert(!IsNull(node)); return IDX(node); }

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
  point_3f nf(normal.x, normal.y, normal.z);
  nf /= 127.0f;
  leaf = PackVoxData(color, PackNormal(nf.x, nf.y, nf.z));
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
    VoxNormal n;

    if (IsLeaf(ch))
    {
      VoxLeaf & leaf = m_leafs[ToIdx(ch)];
      m_leafs.setItemVer(ToIdx(ch), m_curVersion);
      UnpackVoxData(leaf, c, n);
    }
    else
    {
      VoxNode & nd = m_nodes[ToIdx(ch)];
      m_nodes.setItemVer(ToIdx(ch), m_curVersion);
      nd.parent = node;
      nd.flags.selfChildId = i;
      UnpackVoxData(nd.data, c, n);
    }
    
    accCol  += point_3i(c.x, c.y, c.z);

    point_3f nf;
    UnpackNormal(n, nf.x, nf.y, nf.z);
    accNorm += nf;
    ++count;
  }

  accCol /= count;
  accNorm /= (float)count;
  normalize(accNorm);

  VoxNode & nd = m_nodes[ToIdx(node)];
  m_nodes.setItemVer(ToIdx(node), m_curVersion);
  uchar4 color = make_uchar4(accCol.x, accCol.y, accCol.z, count >= 1 ? 255 : 0);
  nd.data = PackVoxData(color, PackNormal(accNorm.x, accNorm.y, accNorm.z));
  nd.flags.emptyFlag = false;
  std::copy(children, children+8, nd.child);

  return node;
}


struct DynamicSVO::TreeBuilder
{
  DynamicSVO & svo;
  VoxelSource & sampler;

  BuildMode mode;
  int destLevel;
  point_3i pos;

  TreeBuilder(DynamicSVO * svo_, VoxelSource * sampler_) 
    : svo(*svo_) 
    , sampler(*sampler_)
    , mode(BUILD_MODE_GROW)
    , destLevel(8)
  {}

  VoxNodeId BuildRange(int level, const point_3i & p, VoxNodeId node)
  {
    if (mode == BUILD_MODE_GROW && node == FullNode)
      return node;
    if (mode == BUILD_MODE_CLEAR && node == EmptyNode)
      return node;

    int blockSize = 1 << (destLevel - level);
    point_3i blockPos = p * blockSize;
    range_3i rng1(blockPos, blockSize);
    range_3i rng2(pos-sampler.GetPivot(), sampler.GetSize());
    if (!rng1.intersects(rng2))
      return node;

    uchar4 col; 
    char4 n;
    TryRangeResult res = sampler.TryRange(blockPos - pos, blockSize, col, n);
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

void DynamicSVO::BuildRange(int level, const cg::point_3i & pos, BuildMode mode, VoxelSource * src)
{
  ++m_curVersion;
  
  TreeBuilder bld(this, src);
  bld.destLevel = level;
  bld.pos = pos;
  bld.mode = mode;
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
