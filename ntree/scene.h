#pragma once

#include "ntree/nodes.h"
//#include "BrickManager.h"
#include "cu_cpp.h"

class Scene
{
public:
  Scene();
  ~Scene();

  void SetTreeDepth(int depth) { m_treeDepth = depth; }
  int GetTreeDepth() const { return m_treeDepth; }

  void Load(const char * filename);
  void Save(const char * filename);

  ntree::Node * GetRoot() { return &m_root; }

  void AddVolume(cg::point_3i pos, cg::point_3i size, const ntree::ValueType * data);

  std::string GetStats();

  void SetViewSize(point_2i size);
  point_2i GetViewSize() const { return m_viewSize; }

  void Render(uchar4 * img, float * debug);
  
private:
  point_2i m_viewSize;

  ntree::Node m_root;
  int m_treeDepth;
};
