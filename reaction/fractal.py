from __future__ import with_statement
from zgl import *
from PIL import Image
from time import clock

class App(ZglApp):
    def __init__(self):
        ZglApp.__init__(self, OrthoCamera())

        self.viewControl.rect = (-2, -1, 1, 1)
        self.srcTex = Texture2D(Image.open("img/fung.png"))
        self.srcTex.filterLinearMipmap()
        self.srcTex.genMipmaps()
        self.srcTex.setParams( (GL_TEXTURE_MAX_ANISOTROPY_EXT, 16))

        self.fragProg = CGShader('fp40', '''
          uniform sampler2D tex;
          uniform float time;
                     
          float2 cmul(float2 a, float2 b)
          {
            float2 c;
            c.x = a.x*b.x - a.y*b.y;
            c.y = a.x*b.y + a.y*b.x;
            return c;
          }

          float4 main( float2 tc: TEXCOORD0 ) : COLOR 
          { 
            float4 col = float4(1);
            float2 c = tc;
            float2 z = float2(0, 0);
            float t = 0.1*time;
            float2 up = float2(cos(t), sin(t));
            float2 vp = float2(-up.y, up.x);
            for (int i = 0; i < 20; ++i)
            {
              {
                float4 v = tex2D(tex, up*z.x + vp*z.y);
                col = col*(1-v.a) + v*v.a;
              }
              //float2 z2 = cmul(z, z);
              z = cmul(z, z) + c;
            }
            return float4(col);
          }
        ''')
        self.fragProg.tex = self.srcTex
    
    def display(self):
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self.fragProg.time = clock()
        with ctx(self.viewControl.with_vp, self.fragProg):
            drawQuad(self.viewControl.rect)

        glutSwapBuffers()

if __name__ == "__main__":
  viewSize = (800, 600)
  zglInit(viewSize, "hello")

  glutSetCallbacks(App())

  #wglSwapIntervalEXT(0)
  glutMainLoop()
