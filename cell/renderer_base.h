#pragma once

#include "svorenderer.h"
#include "shader.h"
#include "rdd.h"

class RendererBase : public ISVORenderer
{
protected:
  // TODO: getters and setters
  SVOData * m_svo;
  point_2i m_viewSize;

  point_3f m_pos;
  point_3f m_dir;
  point_3f m_up;

  float m_fov;

  SimpleShader m_shader;

  AlignedArray<Color32, 4> m_colorBuf;

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
    m_viewSize = point_2i(width, height);
    m_colorBuf.resize(width*height);
  }

  virtual point_2i GetResolution() const { return m_viewSize; }

  virtual void SetFOV(float fov) { m_fov = fov; }

protected:
  void InitRayDir(RayDirData & res)
  {
    point_3f vfwd = cg::normalized(m_dir);
    point_3f vright = cg::normalized(vfwd ^ m_up);
    point_3f vup = vright ^ vfwd;

    float da = tan(cg::grad2rad(m_fov / 2)) / m_viewSize.x;

    res.du = 2 * vright *da;
    res.dv = -2 * vup * da;
    res.dir0 = vfwd - res.du*m_viewSize.x/2 - res.dv*m_viewSize.y/2;
  }
};
