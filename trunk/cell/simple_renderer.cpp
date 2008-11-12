#include "stdafx.h"
#include "svorenderer.h"
#include "trace_utils.h"
#include "shader.h"

struct TraceResult
{
  VoxNodeId node;
  int child;
  float t;
};

struct RayDirData
{
  point_3f dir0, du, dv;
};

class RendererBase : public ISVORenderer
{
protected:
  // TODO: getters and setters
  SVOData * m_svo;
  point_2i m_viewRes;

  point_3f m_pos;
  point_3f m_dir;
  point_3f m_up;

  float m_fov;

  SimpleShader m_shader;

  std::vector<Color32> m_colorBuf;
  

public:
  RendererBase() : m_svo(NULL), m_up(0, 0, 1), m_fov(70.0) { SetResolution(640, 480); }
  virtual ~RendererBase() {}
  
  virtual void SetScene(SVOData * svo) { m_svo = svo; }
  
  virtual void SetViewPos(const point_3f & pos) 
  { 
    m_pos = pos; 
    m_shader.viewerPos = pos;
    m_shader.lightPos = pos;
  }
  virtual void SetViewDir(const point_3f & dir) { m_dir = dir; }
  virtual void SetViewUp (const point_3f & up) { m_up = up; }
  
  virtual void SetResolution(int width, int height) 
  { 
    m_viewRes = point_2i(width, height);
    m_colorBuf.resize(width*height);
  }

  virtual point_2i GetResolution() const { return m_viewRes; }

  virtual void SetFOV(float fov) { m_fov = fov; }

protected:
  void InitRayDir(RayDirData & res)
  {
    point_3f vfwd = cg::normalized(m_dir);
    point_3f vright = cg::normalized(vfwd ^ m_up);
    point_3f vup = vright ^ vfwd;

    float da = tan(cg::grad2rad(m_fov / 2)) / m_viewRes.x;

    res.du = 2 * vright *da;
    res.dv = -2 * vup * da;
    res.dir0 = vfwd - res.du*m_viewRes.x/2 - res.dv*m_viewRes.y/2;
  }
};


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
