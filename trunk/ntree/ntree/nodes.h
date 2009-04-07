#pragma once

#include "gpu_nodes.h"

namespace ntree
{

inline bool operator == (const ValueType & a, const ValueType & b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

class Node
{
public:
  enum NodeType {Grid, Brick, Const};

  Node() : m_type(Const), m_const(DefValue), m_ptr(NULL) {}
  ~Node() { MakeConst(DefValue); }

  NodeType GetType() const { return m_type; }

  void MakeConst(ValueType v)
  {
    if (m_type == Brick)
      delete [] brickPtr();
    if (m_type == Grid)
      delete [] gridPtr();
    m_ptr = NULL;
    m_type = Const;
    m_const = v;
  }

  void MakeBrick()
  {
    if (m_type == Brick)
      return;
    Assert(m_type == Const);
    ValueType * data = new ValueType[BrickSize3];
    std::fill(data, data + BrickSize3, m_const);
    m_type = Brick;
    m_ptr = data;
  }

  void MakeGrid()
  {
    if (m_type == Grid)
      return;
    Assert(m_type == Const);
    m_type = Grid;
    m_ptr = new Node[GridSize3];
    for (int i = 0; i < GridSize3; ++i)
      gridPtr()[i].m_const = m_const;
  }

  Node & child(const point_3i & p) { return child(ch2ofs(p)); }
  Node & child(int i) { return gridPtr()[i]; }
  ValueType * brickPtr() { Assert(m_type == Brick); return static_cast<ValueType*>(m_ptr); }

  void Shrink(bool deep)
  {
    if (m_type == Brick)
    {
      bool allSame = true;
      for (int i = 1; i < BrickSize3 && allSame; ++i)
        allSame = (brickPtr()[i] == brickPtr()[0]);
      if (allSame)
        MakeConst(brickPtr()[0]);
    }
    else if (m_type == Grid)
    {
      Node * p = gridPtr();
      bool allConst = true;
      for (int i = 0; i < GridSize3; ++i, ++p)
      {
        if (deep) 
          p->Shrink(true);
        allConst = allConst && (p->m_type == Const);
      }
      if (!allConst)
        return;

      p = gridPtr();
      ValueType c = p->m_const;
      bool allSame = true;
      for (int i = 1; i < GridSize3 && allSame; ++i, ++p)
        allSame = (p->m_const == c);
      if (allSame)
        MakeConst(c);
    }
  }
private:
  NodeType m_type;
  ValueType m_const;
  void * m_ptr;

  Node * gridPtr() { Assert(m_type == Grid); return static_cast<Node*>(m_ptr); }

  static int ch2ofs(const point_3i & p) { return (p.z * GridSize + p.y) * GridSize + p.x; }
};

}