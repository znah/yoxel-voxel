#pragma once

#include "ntree.h"

struct Scene
{
  struct AlphaTreeTraits
  {
    typedef uint8 ValueType;
    static const int GridSize = 8;
    static const int BrickSize = 4;
    static ValueType DefValue() { return 0; }
  };


  typedef NTree<AlphaTreeTraits> AlphaTree;

  AlphaTree::Node m_alphaRoot;
};
