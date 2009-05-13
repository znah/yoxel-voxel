#pragma once

namespace ntree
{

struct TreeStat
{
  int gridCount;
  int brickCount;
  int constCount;
};

inline bool operator == (const ntree::TreeStat & a, const ntree::TreeStat & b)
{
  return a.gridCount == b.gridCount && a.brickCount == b.brickCount && a.constCount == b.constCount;
}

inline std::ostream & operator << (std::ostream & os, const ntree::TreeStat & stat)
{
  os << "( g: " << stat.gridCount << ", b: " << stat.brickCount << ", c: " << stat.constCount << " )";
  return os;
}

template <class Traits>
class NTree
{
public:
  typedef typename Traits::ValueType ValueType;
  static const int GridSize = Traits::GridSize;
  static const int GridSize3 = GridSize*GridSize*GridSize;
  static const int BrickSize = Traits::BrickSize;
  static const int BrickSize3 = BrickSize*BrickSize*BrickSize;

  class View;
  struct Node;
  
  NTree() { AdjustDepth(1024); }
  ~NTree() {}

  void AdjustDepth(int maxSceneSize);
 
  int GetDepth() const { return m_depth; }
  int GetExtent() const { return m_extent; }

  TreeStat GatherStat() const;

  template <class Proc>
  void WalkTree(Proc & proc)
  {
    WalkNode(m_root, point_3i(), m_extent, proc);
  }
private:
  int m_depth;
  int m_extent;
  Node m_root;

  void AccumStat(const Node & node, TreeStat & res) const;
  template <class Proc>
  void WalkNode(Node & node, point_3i pos, int sz, Proc & proc);
};

template <class Traits>
struct NTree<Traits>::Node : public noncopyable
{
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
    for (int i = 0; i < GridSize3; ++i)
      AccumStat(node.grid[i], res);
}

template <class Traits>
template <class Proc>
void NTree<Traits>::WalkNode(Node & node, point_3i pos, int sz, Proc & proc)
{
  Assert(sz >= BrickSize);
  range_3i nodeRange(pos, sz);

  Node::Type type = node.GetType();
  if (type == Node::Grid)
  {
    Assert(sz > BrickSize);
    if (!proc.enterGrid(node, nodeRange))
      return;
    int sz2 = sz / GridSize;
    for (walk_3 i(GridSize); !i.done(); i.next())
      WalkNode(node.grid[i.flat()], pos + i.pos()*sz2, sz2, proc);
    proc.exitGrid(node, nodeRange);
  }
  else if (type == Node::Const)
  {
    proc.enterConst(node, nodeRange);
    if (node.GetType() != Node::Const)
      WalkNode(node, pos, sz, proc);
  }
  else // brick
  {
    Assert(sz == BrickSize);
    proc.enterBrick(node, nodeRange);
  }
}

} // namespace ntree
