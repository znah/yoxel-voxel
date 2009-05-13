#include "stdafx.h"
#include "ntree.h"

struct AlphaTraits
{
  typedef uint8 ValueType;
  static const int GridSize  = 8;
  static const int BrickSize = 4;
  static ValueType DefValue() { return 0; }
};
typedef ntree::NTree<AlphaTraits> AlphaTree;

BOOST_AUTO_TEST_CASE( NTree_depth_test )
{
  AlphaTree tree;
  tree.AdjustDepth(1);
  BOOST_CHECK_EQUAL( tree.GetDepth(), 0 );
  tree.AdjustDepth(4);
  BOOST_CHECK_EQUAL( tree.GetDepth(), 0 );
  tree.AdjustDepth(5);
  BOOST_CHECK_EQUAL( tree.GetDepth(), 1 );
  tree.AdjustDepth(32);
  BOOST_CHECK_EQUAL( tree.GetDepth(), 1 );
  tree.AdjustDepth(1024);
  BOOST_CHECK_EQUAL( tree.GetDepth(), 3 );
}

template<class Traits>
void CheckViewRange(typename ntree::NTree<Traits>::View & view, ntree::NTree<Traits> & tree, range_3i range)
{
  view.Attach(tree, range);
  range_3i vrng = view.range();
  BOOST_CHECK(vrng.contains(range));
  BOOST_CHECK(vrng.p1 % Traits::BrickSize == point_3i(0, 0, 0));
  BOOST_CHECK(vrng.p2 % Traits::BrickSize == point_3i(0, 0, 0));
}

BOOST_AUTO_TEST_CASE( NTree_view_size_test )
{
  AlphaTree tree;
  AlphaTree::View view;
  
  CheckViewRange(view, tree, range_3i(point_3i(0, 0, 0), point_3i(4, 4, 4)));
  CheckViewRange(view, tree, range_3i(point_3i(1, 1, 0), point_3i(5, 5, 4)));
  CheckViewRange(view, tree, range_3i(point_3i(0, 0, 0), point_3i(8, 8, 4)));
  CheckViewRange(view, tree, range_3i(point_3i(0, 3, 3), point_3i(15, 15, 10)));
}

BOOST_AUTO_TEST_CASE( NTree_view_edit_test )
{
  AlphaTree tree;
  AlphaTree::View view1, view2;
  view1.Attach(tree, range_3i(point_3i(10, 10, 10), point_3i(20, 20, 20)));
  view2.Attach(tree, range_3i(point_3i(15, 15, 15), point_3i(25, 25, 25)));

  point_3i p(17, 17, 17);
  point_3i p1 = p - view1.range().p1;
  point_3i p2 = p - view2.range().p1;
  BOOST_CHECK_EQUAL(view1.data()[p1], AlphaTraits::DefValue());
  BOOST_CHECK_EQUAL(view2.data()[p2], AlphaTraits::DefValue());
  view1.data()[p1] = 123;
  view1.Commit();
  view2.Update();
  BOOST_CHECK_EQUAL(view2.data()[p2], 123);
}

BOOST_AUTO_TEST_CASE( NTree_shrink_test)
{
  const ntree::TreeStat empty = {0, 0, 1};
  
  AlphaTree tree;
  AlphaTree::View view;
  view.Attach(tree, range_3i( point_3i(100, 150, 200), point_3i(120, 160, 230) ));
  BOOST_CHECK_EQUAL(tree.GatherStat(), empty);
  view.data()[point_3i(5, 7, 8)] = 100;
  view.Commit();
  BOOST_CHECK_EQUAL(tree.GatherStat().brickCount, 1);
  view.data()[point_3i(0, 0, 0)] = 45;
  view.Commit();
  BOOST_CHECK_EQUAL(tree.GatherStat().brickCount, 2);
  view.data()[point_3i(0, 0, 1)] = 12;
  view.Commit();
  BOOST_CHECK_EQUAL(tree.GatherStat().brickCount, 2);
  BOOST_TEST_MESSAGE("TreeStat: " << tree.GatherStat());

  view.data()[point_3i(5, 7, 8)] = 0;
  view.data()[point_3i(0, 0, 0)] = 0;
  view.data()[point_3i(0, 0, 1)] = 0;
  view.Commit();
  BOOST_CHECK_EQUAL(tree.GatherStat(), empty);
}
