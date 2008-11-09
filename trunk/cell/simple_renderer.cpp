#include "stdafx.h"
#include "svorenderer.h"


class SimpleRenderer : public ISVORenderer
{
private:
  SVOData * m_svo;
  point_2i m_viewRes;

  point_3f m_pos;
  cprf m_rotation;

  std::vector<Color32> m_colorBuf;
  

public:
  SimpleRenderer() : m_svo(NULL) { SetResolution(640, 480); }
  virtual ~SimpleRenderer() {}
  
  virtual void SetScene(SVOData * svo) { m_svo = svo; }
  
  virtual void SetViewPos(const point_3f & pos) { m_pos = pos; }
  virtual void SetViewDir(const cprf & rotation) { m_rotation = rotation; }
  
  virtual void SetResolution(int width, int height) 
  { 
    m_viewRes = point_2i(width, height);
    m_colorBuf.resize(width*height);
  }

  virtual point_2i GetResolution() const { return m_viewRes; }

  virtual const Color32 * RenderFrame();
};

shared_ptr<ISVORenderer> CreateSimpleRenderer()
{
  return shared_ptr<ISVORenderer>(new SimpleRenderer);
}




const Color32 * SimpleRenderer::RenderFrame()
{
  if (m_svo == NULL)
    return NULL;

  

  


}
