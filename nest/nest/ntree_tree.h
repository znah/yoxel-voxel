#pragma once

namespace ntree
{

template <class Traits>
class NTree
{
public:
  class View;
  
  NTree() { AdjustDepth(1024); }
  ~NTree() {}

  void AdjustDepth(int maxSceneSize)
  {
    m_depth = 0;
    m_extent = Traits::BrickSize;
    while (m_extent < maxSceneSize)
    {
      m_extent *= Traits::GridSize;
      ++m_depth;
    }
  }

  int GetDepth() const { return m_depth; }
  int GetExtent() const { return m_extent; }

  std::string GetStats() const;
private:
  struct Node;
  int m_depth;
  int m_extent;
  Node m_root;
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
  Node * nodes;
  ValueType * brick;

  Node() : constVal(Traits::DefValue()), nodes(NULL), brick(NULL) {}
  ~Node() { MakeConst(Traits::DefValue()); }

  void MakeConst(ValueType v)
  {
    if (nodes != NULL) { delete [] nodes; nodes = NULL; }
    if (brick != NULL) { delete [] brick; brick = NULL; }
    constVal = v;
  }

  void MakeGrid()
  {
    Assert(brick == NULL);
    if (nodes != NULL)
      return;
    nodes = new Node[GridSize3];
    for (Node * p = nodes; p != nodes + GridSize3; ++p)
      p->constVal = constVal;
  }

  void MakeBrick()
  {
    Assert(nodes == NULL);
    if (brick != NULL)
      return;
    brick = new ValueType[BrickSize3];
    std::fill(brick, brick + BrickSize3, constVal);
  }
};

}