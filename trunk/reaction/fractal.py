from __future__ import with_statement
from zgl import *
from PIL import Image

class App(ZglApp):
    def __init__(self):
        ZglApp.__init__(self, OrthoCamera())

        self.srcTex = Texture2D(Image.open("img/fung.png"))

        self.fragProg = CGShader('fp40', '''
          uniform sampler2D tex;
          float4 main( float2 tc: TEXCOORD0 ) : COLOR 
          { 
            float4 c = tex2D(tex, tc);
            c.rgb *= c.a;
            return c;
          }
        ''')
        self.fragProg.tex = self.srcTex
    
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
