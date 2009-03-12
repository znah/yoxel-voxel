#pragma once

#include "nodes.h"

namespace ntree
{

class Scene
{
public:
  Scene();
  ~Scene();

  Load(const char * filename);
  Save(const char * filename);

  Node * GetRoot() { return m_root; }

  void BuildTree(ValueType * )

private:
  Node * m_root;
};


}