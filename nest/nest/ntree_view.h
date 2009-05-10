#pragma once

namespace ntree
{

template <class Traits>
class NTree<Traits>::View
{
public:
  typedef typename Traits::ValueType ValueType;

  View() : m_tree(NULL) {}
  ~View() { Unattach(); }

  void Attach(NTree<Traits> & tree, const range_3i & range);
  void Unattach();
  void Update();
  void Commit();

  const array_3d_ref<ValueType> & data() { return m_data; }
  range_3i & range() { return m_range; }

private:
  static const int GridSize = Traits::GridSize;
  static const int GridSize3 = GridSize*GridSize*GridSize;
  static const int BrickSize = Traits::BrickSize;
  static const int BrickSize3 = BrickSize*BrickSize*BrickSize;

  NTree<Traits> * m_tree;
  array_3d<ValueType> m_data;
  range_3i m_brickRange;
  range_3i m_range;

  void FetchNode(const Node & node, point_3i pos, int sz);
};

template<class Traits>
void NTree<Traits>::View::Attach(NTree<Traits> & tree, const range_3i & range)
{
  Unattach();
  m_tree = &tree;
  
  point_3i bs = point_3i(BrickSize-1, BrickSize-1, BrickSize-1);
  m_brickRange = range_3i(range.p1/BrickSize, (range.p2 + bs) / BrickSize);
  m_range = range_3i(m_brickRange.p1 * BrickSize, m_brickRange.p2 * BrickSize);
  m_data.resize(m_range.size());
  Update();
}

template<class Traits>
void NTree<Traits>::View::Unattach()
{
  m_tree = NULL;
}

template<class Traits>
void NTree<Traits>::View::FetchNode(const Node & node, point_3i pos, int sz)
{
  range_3i nodeRange(pos, sz);
  if (!intersect(m_range, nodeRange))
    return;

  if (node.nodes != NULL)
  {
    int sz2 = sz / GridSize;
    for (walk_3 i(GridSize); !i.done(); i.next())
      FetchNode(node.nodes[i.flat()], pos + i.pos()*sz2, sz2);
  }
  else if (node.brick == NULL)
  {
    range_3i dstRange(pos - m_range.p1, sz);
    range_3i arrayRange(point_3i(), m_data.extent());
    dstRange &= arrayRange;
    fill(m_data, dstRange, node.constVal);
  }
  else
  {
    //Assert(sz == BrickSize);

  }
}

template<class Traits>
void NTree<Traits>::View::Update()
{
  Assert(m_tree != NULL);
  FetchNode(m_tree->m_root, point_3i(), m_tree->m_extent);



}

template<class Traits>
void NTree<Traits>::View::Commit()
{

}

}