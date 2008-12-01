#pragma once

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "DynamicSVO.h"
#include "SVORenderer.h"

class Demo
{
public:
  Demo();
  ~Demo();

  void Idle();
  void Key(unsigned char key, int x, int y);
  void Display();
  void Resize(int width, int height);

private:
  DynamicSVO m_svo;
  SVORenderer m_renderer;

  GLuint m_texId;
  GLuint m_pboId;
  bool m_pboNeedUnreg;

  point_2i m_viewSize;
};
