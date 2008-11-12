#include "stdafx.h"
#include "renderer_base.h"

#include <boost/thread/thread.hpp>


class PPURendererBase : public RendererBase
{
protected:
  bool RecTrace(VoxNodeId nodeId, point_3f t1, point_3f t2, const uint dirFlags, TraceResult & res) const
  {
    if (IsNull(nodeId) || minCoord(t2) <= 0)
      return false;

    const VoxNode & node = (*m_svo)[nodeId];
    int ch = FindFirstChild(t1, t2);
    while (true)
    {
      if (GetLeafFlag(node.flags, ch^dirFlags))
      {
        res.node = nodeId;
        res.child = ch^dirFlags;
        res.t = maxCoord(t1);
        return true;
      }

      if (RecTrace(node.child[ch^dirFlags], t1, t2, dirFlags, res))
        return true;

      if (!GoNext(ch, t1, t2))
        return false;
    }
  }

  void RenderRect(const RayDirData & rdd, const point_2i & start, const point_2i & size)
  {
    point_2i end = start + size;
    end.x = min(end.x, m_viewSize.x);
    end.y = min(end.y, m_viewSize.y);

    for (int y = start.y; y < end.y; ++y)
    {
      for (int x = start.x; x < end.x; ++x)
      {
        int offs = y*m_viewSize.x + x;
        m_colorBuf[offs] = Color32(0, 0, 0, 0);

        point_3f dir = normalized(rdd.dir0 + rdd.du*x + rdd.dv*y);
        AdjustDir(dir);
        point_3f t1, t2;
        uint dirFlags;
        if (!SetupTrace(m_pos, dir, t1, t2, dirFlags))
          continue;

        TraceResult res;
        if (!RecTrace(m_svo->GetRoot(), t1, t2, dirFlags, res))
          continue;

        m_colorBuf[offs] = m_shader.Shade((*m_svo)[res.node].child[res.child], dir, res.t);
      }
    }
  }
};

class SimpleRenderer : public PPURendererBase
{
public:
  virtual const Color32 * RenderFrame()
  {
    if (m_svo == NULL)
      return NULL;

    RayDirData rdd;
    InitRayDir(rdd);
    RenderRect(rdd, point_2i(0, 0), m_viewSize);
    return &m_colorBuf[0];
  }
};

shared_ptr<ISVORenderer> CreateSimpleRenderer()
{
  return shared_ptr<ISVORenderer>(new SimpleRenderer);
}




class TreadedRenderer : public PPURendererBase
{
public:
  virtual const Color32 * RenderFrame();
private:
  struct RenderThreadData;
};

shared_ptr<ISVORenderer> CreateThreadedRenderer()
{
  return shared_ptr<ISVORenderer>(new TreadedRenderer);
}

struct TreadedRenderer::RenderThreadData
{
  TreadedRenderer * renderer;
  RayDirData rdd;
  point_2i start, size;

  void operator()()
  {
    renderer->RenderRect(rdd, start, size);
  }
};

const Color32 * TreadedRenderer::RenderFrame()
{
  if (m_svo == NULL)
    return NULL;

  RayDirData rdd;
  InitRayDir(rdd);

  const int ThreadNum = 4;
  int ystep = m_viewSize.y / ThreadNum;

  boost::thread_group threads;
  for (int i = 0; i < ThreadNum; ++i)
  {
    RenderThreadData td;
    td.renderer = this;
    td.rdd = rdd;
    td.start = point_2i(0, ystep*i);
    td.size = point_2i(m_viewSize.x, ystep);
    threads.create_thread(td);
  }
  threads.join_all();
  return &m_colorBuf[0];
};

