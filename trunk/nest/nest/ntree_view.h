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

  template <bool ToTree>
  void CopyNode(Node & node, point_3i pos, int sz);
  template <bool ToBrick>
  void CopyBrick(ValueType * brick, const point_3i & dstPos);
  void ShrinkGrid(Node & node);
  void ShrinkBrick(Node & node);
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
template<bool ToTree>
void NTree<Traits>::View::CopyNode(Node & node, point_3i pos, int sz)
{
  Assert(sz >= BrickSize);
  range_3i nodeRange(pos, sz);
  if (!intersect(m_range, nodeRange))
    return;

  Node::Type type = node.GetType();
  if (type == Node::Grid)
  {
    int sz2 = sz / GridSize;
    for (walk_3 i(GridSize); !i.done(); i.next())
      CopyNode<ToTree>(node.grid[i.flat()], pos + i.pos()*sz2, sz2);
    if (ToTree)
      ShrinkGrid(node);
  }
  else if (type == Node::Const)
  {
    if (ToTree)
    {
      if (sz > BrickSize)
        node.MakeGrid();
      else
        node.MakeBrick();
      CopyNode<true>(node, pos, sz);
    } else {
      range_3i dstRange(pos - m_range.p1, sz);
      range_3i arrayRange(point_3i(), m_data.extent());
      dstRange &= arrayRange;
      fill(m_data, dstRange, node.constVal);
    }
  }
  else // brick
  {
    Assert(sz == BrickSize);
    CopyBrick<ToTree>(node.brick, pos - m_range.p1);
    if (ToTree)
      ShrinkBrick(node);
  }
}

template <class Traits>
template <bool ToBrick>
void NTree<Traits>::View::CopyBrick(ValueType * brick, const point_3i & dstPos)
{
  for (int z = 0; z < BrickSize; ++z)
  for (int y = 0; y < BrickSize; ++y)
  {
    ValueType * view = m_data.data() + m_data.ofs(dstPos + point_3i(0, y, z));
    for (int x = 0; x < BrickSize; ++x, ++brick, ++view)
      ToBrick ? *brick = *view : *view = *brick;
  }
}

template <class Traits>
void NTree<Traits>::View::ShrinkGrid(Node & node)
{
  Assert(node.GetType() == Node::Grid);
  bool allEq = (node.grid[0].GetType() == Node::Const);
  for (int i = 1; i < GridSize3 && allEq; ++i)
  {
    allEq = node.grid[i].GetType() == Node::Const;
    if (allEq)
      allEq = (node.grid[i].constVal == node.grid[0].constVal);
  }
  if (allEq)
    node.MakeConst(node.grid[0].constVal);
}

template <class Traits>
void NTree<Traits>::View::ShrinkBrick(Node & node)
{
  Assert(node.GetType() == Node::Brick);
  bool allEq = true;
  for (int i = 1; i < BrickSize3 && allEq; ++i)
    allEq = (node.brick[i] == node.brick[0]);
  if (allEq)
    node.MakeConst(node.brick[0]);
}


template<class Traits>
void NTree<Traits>::View::Update()
{
  Assert(m_tree != NULL);
  CopyNode<false>(m_tree->m_root, point_3i(), m_tree->m_extent);
}

template<class Traits>
void NTree<Traits>::View::Commit()
{
  Assert(m_tree != NULL);
  CopyNode<true>(m_tree->m_root, point_3i(), m_tree->m_extent);
}

}