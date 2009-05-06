#pragma once

template <class Traits>
struct NTree
{
  typedef typename Traits::ValueType ValueType;
  static const int GridSize = Traits::GridSize;
  static const int GridSize3 = GridSize*GridSize*GridSize;

  static const int BrickSize = Traits::BrickSize;
  static const int BrickSize3 = BrickSize*BrickSize*BrickSize;

  static int calcSceneSize(int depth)
  {
    int sz = 1;
    while (depth--)
      sz *= GridSize;
    return sz * BrickSize;
  }

  class Node
  {
  public:
    enum NodeType {Grid, Brick, Const};

    Node() : m_type(Const), m_const(Traits::DefValue()), m_ptr(NULL) {}
    ~Node() { MakeConst(Traits::DefValue()); }

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

    Node * gridPtr() { Assert(m_type == Grid); return static_cast<Node*>(m_ptr); }
    ValueType * brickPtr() { Assert(m_type == Brick); return static_cast<ValueType*>(m_ptr); }
    Node & child(const point_3i & p) { return child(ch2ofs(p)); }
    Node & child(int i) { return gridPtr()[i]; }
    ValueType GetValue() const { Assert(m_type == Const); return m_const; }

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

    static int ch2ofs(const point_3i & p) { return (p.z * GridSize + p.y) * GridSize + p.x; }
  };

  struct Builder
  {
    int sceneVoxSize;
    Node * treeRoot;
    array_3d_ref<ValueType> src;
    range_3i srcRange;
    point_3i dstOfs;

    range_3i dstRange;

    void build()
    {
      dstRange = range_3i(dstOfs, srcRange.size());
      buildNode(*treeRoot, sceneVoxSize, point_3i());
    }

    void buildNode(Node & node, int nodeVoxSize, const point_3i & nodePos)
    {
      range_3i range(nodePos, nodeVoxSize);
      if (!range.intersects(dstRange))
        return;

      Assert(nodeVoxSize >= BrickSize);
      if (nodeVoxSize == BrickSize)
      {
        node.MakeBrick();
        updateBrick(node.brickPtr(), range);
      }
      else
      {
        node.MakeGrid();
        int sz = nodeVoxSize / GridSize;
        for (walk_3 i(GridSize); !i.done(); ++i)
          buildNode(node.child(i.p), sz, nodePos + (i.p) * sz);
      }
      node.Shrink(false);
    }

    void updateBrick(ValueType * dstBuf, const range_3i & nodeRange)
    {
      array_3d_ref<ValueType> dst( point_3i(BrickSize, BrickSize, BrickSize), dstBuf );
      range_3i updateRange = nodeRange;
      updateRange &= dstRange;
      point_3i upd2node = updateRange.p1 - nodeRange.p1;
      point_3i node2data = nodeRange.p1 - dstRange.p1 + srcRange.p1;
      for (walk_3 i(updateRange.size()); !i.done(); ++i)
      {
        point_3i dstPt = i.p + upd2node;
        point_3i srcPt = dstPt + node2data;
        dst[dstPt] = src[srcPt];
      }
    }
  };

  struct Fetcher
  {
    int sceneVoxSize;
    Node * treeRoot;
    array_3d_ref<ValueType> dst;
    range_3i srcRange;

    void fetch()
    {
      fetchNode(*treeRoot, sceneVoxSize, point_3i());
    }

    void fetchNode(Node & node, int nodeVoxSize, const point_3i & nodePos)
    {
      range_3i range(nodePos, nodeVoxSize);
      if (!range.intersects(srcRange))
        return;

      if (node.GetType() == Node::Brick)
        fetchBrick(node.brickPtr(), range);
      else if (node.GetType() == Node::Const)
        fetchConst(node.GetValue(), range);
      else
      {
        int sz = nodeVoxSize / GridSize;
        for (walk_3 i(GridSize); !i.done(); ++i)
          fetchNode(node.child(i.p), sz, nodePos + (i.p) * sz);
      }
    }

    void fetchBrick(const ValueType * srcBuf, const range_3i & nodeRange)
    {
      array_3d_ref<const ValueType> src( point_3i(BrickSize, BrickSize, BrickSize), srcBuf );
      range_3i updateRange = nodeRange;
      updateRange &= srcRange;
      point_3i upd2node = updateRange.p1 - nodeRange.p1;
      point_3i node2data = nodeRange.p1 - srcRange.p1;
      for (walk_3 i(updateRange.size()); !i.done(); ++i)
      {
        point_3i srcPt = i.p + upd2node;
        point_3i dstPt = srcPt + node2data;
        dst[dstPt] = src[srcPt];
      }
      //printf("%d %d %d\n", nodeRange.p1.x, nodeRange.p1.y, nodeRange.p1.z);
    }

    void fetchConst(ValueType c, const range_3i & nodeRange)
    {
      range_3i updateRange = nodeRange;
      updateRange &= srcRange;
      point_3i upd2data = updateRange.p1 - srcRange.p1;
      for (walk_3 i(updateRange.size()); !i.done(); ++i)
      {
        point_3i dstPt = i.p + upd2data;
        dst[dstPt] = c;
      }
    }

  };
};
