#pragma once

#include "ntree.h"

enum UpdateMode { UPDATE_GROW, UPDATE_CLEAR };

point_3i max(const point_3i & a, const point_3i & b)
{
  return point_3i( cg::max(a.x, b.x), cg::max(a.y, b.y), cg::max(a.z, b.z) );
}

class Scene
{
private:
  struct AlphaTreeTraits
  {
    typedef uint8 ValueType;
    static const int GridSize = 8;
    static const int BrickSize = 4;
    static ValueType DefValue() { return 0; }
  };

  typedef NTree<AlphaTreeTraits> AlphaTree;
  AlphaTree::Node m_alphaRoot;
  int m_alphaSceneSize;
  array_3d<uint8> m_alphaView;

public:
  Scene(int sceneSize)
  {
    int depth = 0;
    while (AlphaTree::calcSceneSize(depth) < sceneSize)
      ++depth;
    m_alphaSceneSize = AlphaTree::calcSceneSize(depth);
  }

  void updateVolume(const array_3d_ref<uint8> & src, point_3i dstPos, UpdateMode mode)
  {
    const point_3i one(1, 1, 1);
    point_3i padSize = src.extent() + point_3i(2, 2, 2);
    m_alphaView.resize( max(m_alphaView.extent(), padSize ));

    AlphaTree::Fetcher fetch;
    fetch.sceneVoxSize = m_alphaSceneSize;
    fetch.treeRoot = &m_alphaRoot;
    fetch.dst = m_alphaView;
    fetch.srcRange = range_3i(dstPos - one, padSize);
    fetch.fetch();

    if (mode == UPDATE_GROW)
    {
      for (walk_3 i(src.extent()); !i.done(); ++i)
        cg::make_max(m_alphaView[i.p + one], src[i.p]);
    }
    else
    {
      for (walk_3 i(src.extent()); !i.done(); ++i)
        cg::make_min(m_alphaView[i.p + one], src[i.p]);
    }

    AlphaTree::Builder build;
    build.sceneVoxSize = m_alphaSceneSize;
    build.treeRoot = &m_alphaRoot;
    build.src = m_alphaView;
    build.srcRange = range_3i(one, src.size());
    build.dstOfs = dstPos;
    build.build();
  }

  void fetchVolume(range_3i range, array_3d<uint8> & dst)
  {
    dst.resize(range.size());
    AlphaTree::Fetcher fetch;
    fetch.sceneVoxSize = m_alphaSceneSize;
    fetch.treeRoot = &m_alphaRoot;
    fetch.dst = dst;
    fetch.srcRange = range;
    fetch.fetch();
  }
};
