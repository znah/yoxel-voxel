#include "stdafx.h"
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "Demo.h"

Demo * demo = NULL;

void mouseButton(int button, int state, int x, int y) { demo->MouseButton(button, state, x, y); }
void mouseMotion(int x, int y) { demo->MouseMotion(x, y); }
void idle() { demo->Idle(); }
void keyDown(unsigned char key, int x, int y) { demo->KeyDown(key, x, y); }
void keyUp(unsigned char key, int x, int y) { demo->KeyUp(key, x, y); }
void display() { demo->Display(); }
void resize(int width, int height) { demo->Resize(width, height); }
void close() { delete demo; demo = 0; }

int main(int argc, char *argv[])
{
  glutInitWindowSize(800, 600);
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutCreateWindow("yoxel-voxel");
  glewInit();

  //glutSpecialFunc(special);
  glutMouseFunc(mouseButton);
  glutMotionFunc(mouseMotion);
  glutIdleFunc(idle);
  glutKeyboardFunc(keyDown);
  glutKeyboardUpFunc(keyUp);
  glutDisplayFunc(display);
  glutReshapeFunc(resize);
  glutCloseFunc(close);  

  demo = new Demo();
  glutSetOption ( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION ) ;
  glutMainLoop();
  
  assert(demo == NULL);

  return 0;
}
