#include "stdafx.h"
#include "renderer_base.h"
#include "trace_utils.h"

class RecRenderer : public RendererBase
{
public:
  virtual const Color32 * RenderFrame();

private:
  bool RecTrace(VoxNodeId nodeId, point_3f t1, point_3f t2, const uint dirFlags, TraceResult & res);
};

shared_ptr<ISVORenderer> CreateRecRenderer()
{
  return shared_ptr<ISVORenderer>(new RecRenderer);
}

const Color32 * RecRenderer::RenderFrame()
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
}

bool RecRenderer::RecTrace(VoxNodeId nodeId, point_3f t1, point_3f t2, const uint dirFlags, TraceResult & res)
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
