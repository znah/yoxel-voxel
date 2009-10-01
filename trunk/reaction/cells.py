from __future__ import with_statement
from zgl import *


class App:
    def __init__(self, viewSize):
        self.viewControl = OrthoCamera()

        self.cellFrag = CGShader("fp40", fileName = "cells.cg")

        self.noiseTex = Texture2D(random.rand(512, 512, 4).astype(float32), format=GL_RGBA_FLOAT32_ATI)
        self.cellFrag.noiseTex = self.noiseTex
        self.cellFrag.noiseSize = self.noiseTex.size


    def resize(self, x, y):
        self.viewControl.resize(x, y)

    def idle(self):
        glutPostRedisplay()
    
    def display(self):
        
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glViewport(0, 0, self.viewControl.viewSize[0], self.viewControl.viewSize[1])

        with ctx(self.viewControl, self.cellFrag):
            drawQuad()

        glutSwapBuffers()

    def keyDown(self, key, x, y):
        if ord(key) == 27:
            glutLeaveMainLoop()
                
    def keyUp(self, key, x, y):
        pass

    def mouseMove(self, x, y):
        self.viewControl.mouseMove(x, y)

    def mouseButton(self, btn, up, x, y):
        self.viewControl.mouseButton(btn, up, x, y)


if __name__ == "__main__":
  viewSize = (800, 600)
  zglInit(viewSize, "hello")

  app = App(viewSize)
  glutSetCallbacks(app)

  #wglSwapIntervalEXT(0)
  glutMainLoop()
