#pragma once

#include <vector>
#include <algorithm>
#include "point.h"
#include "HomoStorage.h"

#include "trace_cu.h"
#include "VoxelSource.h"

enum BuildMode { BUILD_MODE_GROW, BUILD_MODE_CLEAR };

inline bool IsNull(VoxNodeId node) { return node < 0; }

class DynamicSVO
{
private:
  typedef HomoStorage<VoxNode> NODES;
  
  struct TreeBuilder;

  NODES m_nodes;
  VoxNodeId m_root;

  int m_curVersion;

  VoxNodeId CreateNode();
  void DelNode(VoxNodeId nodeId);

  VoxNodeId RecTrace(VoxNodeId node, cg::point_3f t1, cg::point_3f t2, const uint dirFlags, float & t) const;

public:
  explicit DynamicSVO() : m_root(CreateNode()), m_curVersion(0) {}

  void BuildRange(int level, const cg::point_3i & pos, BuildMode mode, VoxelSource * src);

  const NODES & GetNodes() const { return m_nodes; }
  VoxNodeId GetRoot() const { return m_root; }

  int GetNodeCount() const { return m_nodes.count(); }
  int CountChangedPages() const;
  int CountTransfrerSize() const;

  int GetCurVersion() const { return m_curVersion; }

  void Save(std::string filename);
  bool Load(std::string filename);

  float TraceRay(const cg::point_3f & p, cg::point_3f dir) const;
};