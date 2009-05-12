#pragma once

namespace ntree
{

struct TreeStat
{
  int gridCount;
  int brickCount;
  int constCount;
};

template <class Traits>
class NTree
{
public:
  class View;
  
  NTree() { AdjustDepth(1024); }
  ~NTree() {}

  void AdjustDepth(int maxSceneSize);
 
  int GetDepth() const { return m_depth; }
  int GetExtent() const { return m_extent; }

  TreeStat GatherStat() const;
private:
  struct Node;
  int m_depth;
  int m_extent;
  Node m_root;

  void AccumStat(const Node & node, TreeStat & res) const;
};

template <class Traits>
struct NTree<Traits>::Node
{
  typedef typename Traits::ValueType ValueType;
  static const int GridSize = Traits::GridSize;
  static const int GridSize3 = GridSize*GridSize*GridSize;
  static const int BrickSize = Traits::BrickSize;
  static const int BrickSize3 = BrickSize*BrickSize*BrickSize;

  ValueType constVal;
  Node * grid;
  ValueType * brick;

  enum Type { Grid, Brick, Const };

  Node() : constVal(Traits::DefValue()), grid(NULL), brick(NULL) {}
  ~Node() { MakeConst(Traits::DefValue()); }

  Type GetType() const 
  { 
    if (grid != NULL)
      return Grid;
    if (brick != NULL)
      return Brick;
    return Const;
  }

  void MakeConst(ValueType v)
  {
    if (grid != NULL) { delete [] grid; grid = NULL; }
    if (brick != NULL) { delete [] brick; brick = NULL; }
    constVal = v;
  }

  void MakeGrid()
  {
    Assert(brick == NULL);
    if (grid != NULL)
      return;
    grid = new Node[GridSize3];
    for (Node * p = grid; p != grid + GridSize3; ++p)
      p->constVal = constVal;
  }

  void MakeBrick()
  {
    Assert(grid == NULL);
    if (brick != NULL)
      return;
    brick = new ValueType[BrickSize3];
    std::fill(brick, brick + BrickSize3, constVal);
  }
};

template <class Traits>
void NTree<Traits>::AdjustDepth(int maxSceneSize)
{
  m_depth = 0;
  m_extent = Traits::BrickSize;
  while (m_extent < maxSceneSize)
  {
    m_extent *= Traits::GridSize;
    ++m_depth;
  }
}

template <class Traits>
TreeStat NTree<Traits>::GatherStat() const
{
  TreeStat stat = {0, 0, 0};
  AccumStat(m_root, stat);
  return stat;
}

template <class Traits>
void NTree<Traits>::AccumStat(const Node & node, TreeStat & res) const
{
  Node::Type type = node.GetType();
  if (type == Node::Brick)
    ++res.brickCount;
  else if (type == Node::Grid)
    ++res.gridCount;
  else
    ++res.constCount;

  if (type == Node::Grid)
    for (int i = 0; i < Node::GridSize3; ++i)
      AccumStat(node.grid[i], res);
}

}