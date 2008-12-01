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
  cudaGLRegisterBufferObject(m_pboId);
  m_pboNeedUnreg = true;

  m_viewSize = point_2i(width, height);

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

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glutSwapBuffers();
}

