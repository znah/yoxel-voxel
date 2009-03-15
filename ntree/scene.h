#pragma once

#include "ntree/nodes.h"

class Scene
{
public:
  Scene();
  ~Scene();

  void SetTreeDepth(int depth) { m_treeDepth = depth; }
  int GetTreeDepth() const { return m_treeDepth; }

  void Load(const char * filename);
  void Save(const char * filename);

  ntree::NodePtr GetRoot() { return m_root; }

  void AddVolume(cg::point_3i pos, cg::point_3i size, const ntree::ValueType * data);

  std::string GetStats();

  ntree::ValueType TraceRay(const point_3f & p, point_3f dir);
  
private:
  template <class NodeProc>
  void walkTree(NodeProc & proc);

  ntree::NodePtr m_root;
  int m_treeDepth;
};
