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

  ValueType & operator[](const point_3i & p) { return m_data[p]; }

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
  struct TreeProc;

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

template<class Traits>
void NTree<Traits>::View::Update()
{
  Assert(m_tree != NULL);

  TreeProc<false> proc;
  proc.view = this;
  m_tree->WalkTree(proc);
}

template<class Traits>
void NTree<Traits>::View::Commit()
{
  Assert(m_tree != NULL);

  TreeProc<true> proc;
  proc.view = this;
  m_tree->WalkTree(proc);
}

template <class Traits>
template <bool ToTree>
struct NTree<Traits>::View::TreeProc
{
  View * view;
  bool enterGrid(Node & node, const range_3i & range) 
  {
    return intersect(view->m_range, range);
  }
  void exitGrid(Node & node, const range_3i & range) 
  { 
    if (ToTree)
      ShrinkGrid(node); 
  }
  void enterBrick(Node & node, const range_3i & range)
  {
    if (!intersect(view->m_range, range))
      return;
    Assert(range.size() == point_3i(BrickSize, BrickSize, BrickSize));
    view->CopyBrick<ToTree>(node.brick, range.p1 - view->m_range.p1);
    if (ToTree)
      ShrinkBrick(node);
  }

  void enterConst(Node & node, const range_3i & range)
  {
    if (!intersect(view->m_range, range))
      return;
    int sz = range.size().x;
    if (ToTree)
    {
      (sz > BrickSize) ? node.MakeGrid() : node.MakeBrick();
    } else {
      range_3i dstRange(range.p1 - view->m_range.p1, sz);
      range_3i arrayRange(point_3i(), view->m_data.extent());
      dstRange &= arrayRange;
      fill(view->m_data, dstRange, node.constVal);
    }
  }

  void ShrinkGrid(Node & node)
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

  void ShrinkBrick(Node & node)
  {
    Assert(node.GetType() == Node::Brick);
    bool allEq = true;
    for (int i = 1; i < BrickSize3 && allEq; ++i)
      allEq = (node.brick[i] == node.brick[0]);
    if (allEq)
      node.MakeConst(node.brick[0]);
  }
};


}