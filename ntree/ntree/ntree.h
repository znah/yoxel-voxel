#pragma once

#include "nodes.h"

namespace ntree
{

class Scene
{
public:
  Scene();
  ~Scene();

  void SetTreeDepth(int depth) { m_treeDepth = depth; }
  int GetTreeDepth() const { return m_treeDepth; }

  void Load(const char * filename);
  void Save(const char * filename);

  Node * GetRoot() { return m_root; }

  void AddVolume(cg::point_3i pos, cg::point_3i size, const ValueType * data);
  
private:
  template <class NodeProc>
  void walkTree(NodeProc & proc);

  NodePtr m_root;
  int m_treeDepth;
};


}