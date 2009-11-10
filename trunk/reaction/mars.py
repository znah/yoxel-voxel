from __future__ import with_statement
from zgl import *

class App(ZglApp):
    def __init__(self):
        ZglApp.__init__(self, OrthoCamera())


        self.flowTex = RenderTexture(size = (512, 512), format = GL_RGBA_FLOAT16_ATI)
        

        self.fragProg = CGShader('fp40', '''
          uniform sampler flowTex;

          float4 main( float2 tc: TEXCOORD0 ) : COLOR 
          { 
            return tex2D(flowTex, tc); 
          }
        ''')
        self.fragProg = self.flowTex
    
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
