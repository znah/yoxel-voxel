#include "stdafx.h"
#include "Demo.h"
#include <cuda_gl_interop.h>
#include "format.h"

using std::cout;
using std::endl;

Demo::Demo() 
: m_pboNeedUnreg(false)
, m_pos(0.15224274f, 0.50049937f, 0.12925711f)
, m_crs(300.0f)
, m_pitch(-17.0f)
, m_mouseMoving(false)
, m_frameCount(0)
{
  cout << "loading scene ...";
  if (!m_svo.Load("../data/scene.vox"))
    throw std::runtime_error("Can't load scene");
  cout << " OK" << endl;
  
  m_renderer.SetScene(&m_svo);

  m_renderer.SetViewPos(m_pos);
  m_renderer.SetViewDir(CalcViewDir());

  glGenTextures(1, &m_texId);
  glGenBuffers(1, &m_pboId);

  m_lastTime = GetTime();
  m_lastFPSTime = GetTime();
}

double Demo::GetTime() { return glutGet(GLUT_ELAPSED_TIME) / 1000.0; }

Demo::~Demo() 
{
  glDeleteTextures(1, &m_texId);
  glDeleteBuffers(1, &m_pboId);
}


void Demo::Resize(int width, int height)
{
  m_renderer.SetViewSize(width, height);
  
  if (m_pboNeedUnreg)
    cudaGLUnregisterBufferObject(m_pboId);
  
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pboId);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(Color32)*width*height, NULL, GL_STREAM_DRAW);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  cudaGLRegisterBufferObject(m_pboId);
  m_pboNeedUnreg = true;

  CUT_CHECK_ERROR("ttt");

  m_viewSize = point_2i(width, height);

  glBindTexture(GL_TEXTURE_2D, m_texId);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_viewSize.x, m_viewSize.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );


  glViewport(0, 0, width, height);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, 1, 0, 1);

  cout << "resized " << width << "x" << height << endl;
}


void Demo::Idle()
{
  double curTime = GetTime();
  float dt = float(curTime - m_lastTime);
  m_lastTime = curTime;
  
  if (curTime - m_lastFPSTime > 0.5)
  {
    float fps = (float)(m_frameCount / (curTime - m_lastFPSTime));
    glutSetWindowTitle(format("yoxel-voxel -- fps: {0}") % fps);
    m_lastFPSTime = curTime;
    m_frameCount = 0;
  }
  ++m_frameCount;

  point_3f fwdDir = CalcViewDir();
  point_3f upDir(0, 0, 1);
  point_3f rightDir = cg::normalized(fwdDir ^ upDir);
  point_3f dir = fwdDir*m_motionVel.y + rightDir*m_motionVel.x;
  m_pos += dir*dt*0.3; 

  m_renderer.SetViewDir(fwdDir);
  m_renderer.SetViewPos(m_pos);


  void * d_pboPtr = 0;
  cudaGLMapBufferObject(&d_pboPtr, m_pboId); 
  m_renderer.Render(d_pboPtr);
  cudaGLUnmapBufferObject(m_pboId);

  glBindTexture(GL_TEXTURE_2D, m_texId);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pboId);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_viewSize.x, m_viewSize.y, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glBindTexture(GL_TEXTURE_2D, 0);

  glutPostRedisplay();
}

void Demo::KeyDown(unsigned char key, int x, int y)
{
  if (key == 27)
    glutLeaveMainLoop();
  
  if (key == 'a') m_motionVel.x = -1;
  if (key == 'd') m_motionVel.x = 1;
  if (key == 'w') m_motionVel.y = 1;
  if (key == 's') m_motionVel.y = -1;

  if (key == '9') m_renderer.SetDetailCoef( m_renderer.GetDetailCoef() - 0.1f );
  if (key == '0') m_renderer.SetDetailCoef( m_renderer.GetDetailCoef() + 0.1f );

  if (key == '-') m_renderer.SetFOV( m_renderer.GetFOV()*1.1f );
  if (key == '=') m_renderer.SetFOV( m_renderer.GetFOV()*0.9f );

  if (key == '[') m_renderer.SetDither( m_renderer.GetDither()*1.1f );
  if (key == ']') m_renderer.SetDither( m_renderer.GetDither()*0.9f );
}

void Demo::KeyUp(unsigned char key, int x, int y)
{
  if (key == 'a' || key == 'd') 
    m_motionVel.x = 0;
  if (key == 'w' || key == 's') 
    m_motionVel.y = 0;
}


void Demo::MouseButton(int button, int state, int x, int y)
{
  if (button == GLUT_LEFT_BUTTON || state == GLUT_DOWN)
  {
    m_mouseMoving = true;
    m_prevMousePos = point_2i(x, y);
  }
  else
    m_mouseMoving = false;
}

void Demo::MouseMotion(int x, int y)
{
  if (!m_mouseMoving)
    return;

  point_2i dp = point_2i(x, y) - m_prevMousePos;
  m_prevMousePos = point_2i(x, y);
  m_crs = cg::norm360(m_crs - dp.x*0.5f);
  m_pitch = cg::bound(m_pitch - dp.y*0.5f, -89.0f, +89.0f);
}

point_3f Demo::CalcViewDir()
{
  float c = cg::grad2rad(m_crs);
  float p = cg::grad2rad(m_pitch);
  float vcos = cos(p);
  return point_3f(cos(c)*vcos, sin(c)*vcos, sin(p));
}


void Demo::Display()
{
  glClear(GL_COLOR_BUFFER_BIT);

  glColor3f(1, 0.5, 1);

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glEnable(GL_TEXTURE_2D);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  
  glBindTexture(GL_TEXTURE_2D, m_texId);
  
  glBegin(GL_QUADS);
  glTexCoord2f(0, 0); glVertex2f(0, 0);
  glTexCoord2f(1, 0); glVertex2f(1, 0);
  glTexCoord2f(1, 1); glVertex2f(1, 1);
  glTexCoord2f(0, 1); glVertex2f(0, 1);
  glEnd();

  glutSwapBuffers();
}

