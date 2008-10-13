#pragma once

#include <vector>
#include <algorithm>
#include "point.h"
#include "HomoStorage.h"

#include "trace_cu.h"
#include "VoxelSource.h"

enum BuildMode { BUILD_MODE_GROW, BUILD_MODE_CLEAR };

inline bool IsNull(VoxNodeId node) { return node < 0; }
inline bool IsLeaf(VoxNodeId node) { return !IsNull(node) && (node & VOX_LEAF) != 0; }
inline bool IsNode(VoxNodeId node) { return !IsNull(node) && !IsLeaf(node); }

class DynamicSVO
{
private:
  typedef HomoStorage<VoxNode> NODES;
  typedef HomoStorage<VoxLeaf> LEAFS;
  
  struct TreeBuilder;

  NODES m_nodes;
  LEAFS m_leafs;
  VoxNodeId m_root;

  int m_curVersion;

  void DelNode(VoxNodeId node, bool recursive = true);
  VoxNodeId SetLeaf(VoxNodeId node, uchar4 color, char4 normal);
  void FetchChildren(VoxNodeId node, VoxNodeId * dst) const;
  VoxNodeId UpdateChildren(VoxNodeId node, const VoxNodeId * children);

  VoxNodeId RecTrace(VoxNodeId node, cg::point_3f t1, cg::point_3f t2, const uint dirFlags, float & t) const;

public:
  explicit DynamicSVO() : m_root(-1), m_curVersion(0) {}

  void BuildRange(int level, const cg::point_3i & pos, BuildMode mode, VoxelSource * src);

  const NODES & GetNodes() const { return m_nodes; }
  const LEAFS & GetLeafs() const { return m_leafs; }
  VoxNodeId GetRoot() const { return m_root; }

  int GetNodeCount() const { return m_nodes.count() + m_leafs.count(); }
  int CountChangedPages() const;
  int CountTransfrerSize() const;

  int GetCurVersion() const { return m_curVersion; }

  void Save(std::string filename);
  bool Load(std::string filename);

  float TraceRay(const cg::point_3f & p, cg::point_3f dir) const;
};