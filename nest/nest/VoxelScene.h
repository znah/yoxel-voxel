#pragma once
#include "ntree.h"

class VoxelScene
{
public:
  VoxelScene() {}
  ~VoxelScene() {}

private:
  struct AlphaTraits
  {
    typedef uint8 ValueType;
    static const int GridSize  = 8;
    static const int BrickSize = 4;
    static ValueType DefValue() { return 0; }
  };
  typedef ntree::NTree<AlphaTraits> AlphaTree;



};
