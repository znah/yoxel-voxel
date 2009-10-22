from __future__ import with_statement
from zgl import *
from time import clock

class App(ZglApp):
    def __init__(self):
        ZglApp.__init__(self, OrthoCamera())

        self.cellFrag = CGShader("fp40", fileName = "cells.cg")

        self.noiseTex = Texture2D(random.rand(512, 512, 4).astype(float32), format=GL_RGBA_FLOAT32_ATI)
        self.cellFrag.noiseTex = self.noiseTex
        self.cellFrag.noiseSize = self.noiseTex.size
    
    def display(self):
        
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self.cellFrag.time = clock()
        with ctx(self.viewControl.with_vp, self.cellFrag):
            drawQuad()

        glutSwapBuffers()

if __name__ == "__main__":
  viewSize = (800, 600)
  zglInit(viewSize, "hello")

  glutSetCallbacks( App() )

  #wglSwapIntervalEXT(0)
  glutMainLoop()
