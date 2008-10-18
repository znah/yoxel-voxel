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
  write(file, m_root);
}

bool DynamicSVO::Load(std::string filename)
{
  std::ifstream file(filename.c_str(), std::ios::binary);
  m_nodes.load(file);
  read(file, m_root);

  m_curVersion = 0;
  return true;
}

VoxNodeId DynamicSVO::CreateNode()
{
  VoxNodeId nodeId = m_nodes.insert();
  VoxNode & node = m_nodes[nodeId];
  node.flags.pad = 0;
  node.flags.emptyFlag = true;
  node.flags.leafFlags = 0xff;

  node.parent = -1;
  std::find(node.child, node.child+8, EmptyNode);
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

  void SetNodeParent(VoxNodeId dst, VoxNodeId parent, int octant)
  {
    if (IsNull(dst))
      return;
    VoxNode & node = svo.m_nodes[dst];
    node->flags.selfChildId = octant;
    node->parent = parent;
  }

  bool IsInRange(const point_3i & blockPos, int blockSize)
  {
    range_3i rng1(blockPos, blockSize);
    range_3i rng2(pos-sampler.GetPivot(), sampler.GetSize());
    return rng1.intersects(rng2);
  }
  
  // true if changed
  bool UpdateNodeLOD(VoxNode * node)
  {

  }
  
  void BuildRange(int level, const point_3i & p, VoxChild & dstChild, bool & dstLeafFlag)
  {
    if (!dstLeafFlag)
    {
      if (mode == BUILD_MODE_GROW && dstChild == FullNode)
        return;
      if (mode == BUILD_MODE_CLEAR && dstChild == EmptyNode)
        return;
    }

    int blockSize = 1 << (destLevel - level);
    point_3i blockPos = p * blockSize;
    if (!IsInRange(blockPos, blockSize))
      return;

    uchar4 col; 
    char4 n;
    TryRangeResult res = sampler.TryRange(blockPos - pos, blockSize, col, n);

    if (res == ResEmpty && mode == BUILD_MODE_CLEAR)
    {
      dstChild = EmptyNode;
      dstLeafFlag = false;
      return;
    }

    if (res == ResFull && mode == BUILD_MODE_GROW)
    {
      dstChild = FullNode;
      dstLeafFlag = false;
      return;
    }

    if (res == ResSurface && (mode == BUILD_MODE_GROW || !dstLeafFlag))
    {
      // TODO: optimize
      point_3f nf(n.x, n.y, n.z);
      nf /= 127.0;
      dstChild = PackVoxData(col, PackNormal(nf.x, nf.y, nf.z));
      dstLeafFlag = true;
      return;
    }

    if (res != ResGoDown)
      return;

    VoxNode node;
    node.flags = 0;
    bool allNull = true;
    for (walk_3 octant(2, 2, 2); !octant.done(); ++octant)
    {
      //bool leaf
      void BuildRange(int level, const point_3i & p, VoxChild & dstChild, bool & dstLeafFlag)


    }
  }

  VoxNodeId UpdateRange(int level, const point_3i & p, VoxNodeId nodeId)
  {
    int blockSize = 1 << (destLevel - level);
    point_3i blockPos = p * blockSize;
    if (!IsInRange(blockPos, blockSize))
      return nodeId;

    uchar4 col; 
    char4 n;
    TryRangeResult res = sampler.TryRange(blockPos - pos, blockSize, col, n);

    if (res == ResEmpty && mode == BUILD_MODE_CLEAR)
    {
      svo.DelNode(nodeId);
      return EmptyNode;
    }

    if (res == ResFull && mode == BUILD_MODE_GROW)
    {
      svo.DelNode(nodeId);
      return FullNode;
    }

    if (res != ResGoDown)
      return nodeId;

    VoxNode node = svo.m_nodes[nodeId];
    bool changed = false;
    for (walk_3 octant(2, 2, 2); !octant.done(); ++octant)
    {
      uchar chMask = 1 << octant.flat();
      bool leafFlag = node.flags.leafFlags & chMask != 0;
      VoxChild & childRef = node.child[octant.flat()];
      if (!leafFlag && !IsNull(childRef))
      {
        childRef = UpdateRange(level+1, p*2+octant.p, childRef);
        if (IsNull(childRef))
          changed = true;
        continue;
      }

      VoxChild newChild = childRef;
      bool newLeafFlag = leafFlag;
      BuildRange(level+1, p*2+octant.p, newChild, newLeafFlag);
      if (newChild == childRef && newLeafFlag == leafFlag)
        continue;
      childRef = newChild;
      leafFlag = newLeafFlag;

      node.flags.leafFlags &= ~chMask;
      if (leafFlag)
        node.flags.leafFlags |= chMask;

      if (!leafFlag)
        SetNodeParent(childRef, nodeId, octant.flat());

      changed = true;        
    }
    
    if (UpdateNodeLOD(node))
      changed = true;

    if (changed)
    {
      svo.m_nodes[nodeId] = node;
      svo.m_nodes.setItemVer(nodeId, svo.GetCurVersion());
    }
    return nodeId;
  }
};

void DynamicSVO::BuildRange(int level, const cg::point_3i & pos, BuildMode mode, VoxelSource * src)
{
  ++m_curVersion;
  
  TreeBuilder bld(this, src);
  bld.destLevel = level;
  bld.pos = pos;
  bld.mode = mode;
  m_root = bld.UpdateRange(0, point_3i(0, 0, 0), m_root);
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
