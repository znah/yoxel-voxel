#pragma once

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "DynamicSVO.h"
#include "SVORenderer.h"
#include "builders.h"

class Demo
{
public:
  Demo();
  ~Demo();

  void Idle();
  void KeyUp(unsigned char key, int x, int y);
  void KeyDown(unsigned char key, int x, int y);
  void Display();
  void Resize(int width, int height);
  void MouseButton(int button, int state, int x, int y);
  void MouseMotion(int x, int y);

private:
  DynamicSVO m_svo;
  SVORenderer m_renderer;

  GLuint m_texId;
  GLuint m_pboId;
  bool m_pboNeedUnreg;

  point_2i m_viewSize;
  
  point_3f m_pos;
  float m_crs, m_pitch;
  bool m_mouseMoving;
  point_2i m_prevMousePos;

  point_3f CalcViewDir();
  point_3f m_motionVel;

  double GetTime();

  double m_lastTime;
  double m_lastFPSTime;
  int m_frameCount;

  enum EditAction { EditNone, EditGrow, EditClear };
  EditAction m_editAction;
  double m_lastEditTime;
  void DoEdit(const point_3f & fwdDir);
};
