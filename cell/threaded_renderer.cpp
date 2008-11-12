#include "stdafx.h"
#include "renderer_base.h"

#include <boost/thread/thread.hpp>

class TreaderRenderer : public RendererBase
{
public:
  virtual const Color32 * RenderFrame();
};

shared_ptr<ISVORenderer> CreateThreadedRenderer()
{
  return shared_ptr<ISVORenderer>(new TreaderRenderer);
}


const Color32 * TreaderRenderer::RenderFrame()
{
  if (m_svo == NULL)
    return NULL;

  RayDirData rdd;
  InitRayDir(rdd);
  for (int y = 0; y < m_viewRes.y; ++y)
  {
    for (int x = 0; x < m_viewRes.x; ++x)
    {
      int offs = y*m_viewRes.x + x;
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
  return &m_colorBuf[0];
};

