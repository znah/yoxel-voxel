#include "stdafx.h"
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "Demo.h"

Demo * demo = NULL;

void idle() { demo->Idle(); }
void key(unsigned char key, int x, int y) { demo->Key(key, x, y); }
void display() { demo->Display(); }
void resize(int width, int height) { demo->Resize(width, height); }
void close() { delete demo; demo = 0; }

int main(int argc, char *argv[])
{
  glutInitWindowSize(640,480);
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutCreateWindow("yoxel-voxel");
  glewInit();

  //glutSpecialFunc(special);
  glutIdleFunc(idle);
  glutKeyboardFunc(key);
  glutDisplayFunc(display);
  glutReshapeFunc(resize);
  glutCloseFunc(close);  

  demo = new Demo();
  glutSetOption ( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION ) ;
  glutMainLoop();
  
  assert(demo == NULL);

  return 0;
}