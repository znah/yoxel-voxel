#pragma once

#include "HomoStorage.h"
#include "vox_node.h"
#include "VoxelSource.h"

enum BuildMode { BUILD_MODE_GROW, BUILD_MODE_CLEAR };

struct TraceResult
{
  float t;
  VoxNode node;
};

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

  void RecNodeCount(VoxNodeId node, int level, std::vector<int> & res) const;

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

  bool TraceRay(const cg::point_3f & p, cg::point_3f dir, TraceResult & res ) const;

  std::vector<int> GetNodeCountByLevel() const;
};