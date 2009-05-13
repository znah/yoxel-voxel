#pragma once
#include "ntree.h"

class VoxelScene
{
public:
  VoxelScene() {}
  ~VoxelScene() {}



  template <class Func>
  void AddSurface(Func & func);

private:
  struct AlphaTraits
  {
    typedef uint8 ValueType;
    static const int GridSize  = 8;
    static const int BrickSize = 4;
    static ValueType DefValue() { return 0; }
  };
  typedef ntree::NTree<AlphaTraits> AlphaTree;

  AlphaTree m_alphaTree;
};
