#include "stdafx.h"
#include <GL/glew.h>
#include <GL/freeglut.h>


void idle()
{
  glutPostRedisplay();
}

void key(unsigned char key, int x, int y)
{
  if (key == 27)
    glutLeaveMainLoop();
  else
    return;

  glutPostRedisplay();
}

void display(void)
{
  //const double t = glutGet(GLUT_ELAPSED_TIME) / 1000.0;

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glutSwapBuffers();
}


int main(int argc, char *argv[])
{
  glutInitWindowSize(640,480);
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutCreateWindow("yoxel-voxel");
  glewInit();

  //glutReshapeFunc(resize);
  glutDisplayFunc(display);
  glutKeyboardFunc(key);
  //glutSpecialFunc(special);
  glutIdleFunc(idle);

  glutSetOption ( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION ) ;
  glutMainLoop();

  return 0;
}