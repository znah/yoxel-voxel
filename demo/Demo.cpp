#include "stdafx.h"
#include "Demo.h"
#include <cuda_gl_interop.h>

using std::cout;
using std::endl;

Demo::Demo() : m_pboNeedUnreg(false)
{
  cout << "loading scene ...";
  if (!m_svo.Load("../data/scene.vox"))
    throw std::runtime_error("Can't load scene");
  cout << " OK" << endl;
  
  m_renderer.SetScene(&m_svo);

  m_renderer.SetViewPos(point_3f(1, 1, 1));
  m_renderer.SetViewDir(point_3f(-1, -1, -2));

  glGenTextures(1, &m_texId);
  glGenBuffers(1, &m_pboId);
}

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
  void * d_pboPtr = 0;
  cudaGLMapBufferObject(&d_pboPtr, m_pboId); 
  m_renderer.Render(d_pboPtr);
  cudaGLUnmapBufferObject(m_pboId);

  glBindTexture(GL_TEXTURE_2D, m_texId);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pboId);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_viewSize.x, m_viewSize.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  glutPostRedisplay();
}

void Demo::Key(unsigned char key, int x, int y)
{
  if (key == 27)
    glutLeaveMainLoop();
  else
    return;

  glutPostRedisplay();
}

void Demo::Display()
{
  //const double t = glutGet(GLUT_ELAPSED_TIME) / 1000.0;

  glClear(GL_COLOR_BUFFER_BIT);

  glColor3f(1, 0.5, 1);

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glEnable(GL_TEXTURE_2D);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  
  glBindTexture(GL_TEXTURE_2D, m_texId);
  
  glBegin(GL_QUADS);

  glTexCoord2f(0, 0);
  glVertex2f(0, 0);

  glTexCoord2f(1, 0);
  glVertex2f(1, 0);

  glTexCoord2f(1, 1);
  glVertex2f(1, 1);

  glTexCoord2f(0, 1);
  glVertex2f(0, 1);

  glEnd();


  glutSwapBuffers();
}

