from __future__ import with_statement
from zgl import *

class App(ZglApp):
    def __init__(self):
        ZglApp.__init__(self, OrthoCamera())

        self.fragProg = CGShader('fp40', '''
          float4 main( float2 tc: TEXCOORD0 ) : COLOR 
          { 
            return float4(tc, 0, 1); 
          }
        ''')
    
    def display(self):
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        with ctx(self.viewControl.with_vp, self.fragProg):
            drawQuad()

        glutSwapBuffers()

if __name__ == "__main__":
  viewSize = (800, 600)
  zglInit(viewSize, "hello")

  glutSetCallbacks(App())

  #wglSwapIntervalEXT(0)
  glutMainLoop()
