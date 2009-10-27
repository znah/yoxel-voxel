from __future__ import with_statement
from zgl import *
from PIL import Image
from time import clock

class App(ZglApp):
    def __init__(self):
        ZglApp.__init__(self, OrthoCamera())

        self.viewControl.rect = (-2, -1, 1, 1)
        self.srcTex = Texture2D(Image.open("img/lines.png"))
        self.srcTex.filterLinearMipmap()
        self.srcTex.genMipmaps()
        self.srcTex.setParams( (GL_TEXTURE_MAX_ANISOTROPY_EXT, 16))
        #self.srcTex.setParams( *Texture2D.ClampToEdge )

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
            float4 col = float4(0);
            
            float2 c = tc;
            float2 z = float2(0, 0);
            float t = 0.2*time;
            float s = 1.0 + sin(5*time)*0.05;
            float2 up = float2(cos(t), sin(t));//*s;
            float2 vp = float2(-up.y, up.x);
            
            //float2 c = float2(-0.726895347709114071439, 0.188887129043845954792);// + up*0.01;
            //float2 z = tc;

            for (int i = 0; i < 30; ++i)
            {
              //float2 z2 = cmul(z, z);
              z = cmul(z, z) + c;
              {
                float2 p = z;
                p = up*p.x + vp*p.y;
                float fade = min(1, 2/(abs(p.x)+abs(p.y)));
                if (isnan(fade))
                  fade = 0;
                p += float2(0.5, 0.5) + float2(i*0.34, i*0.12);
                
                float4 v = tex2D(tex, p) * fade;
                float2 dx = abs(ddx(p));
                float2 dy = abs(ddy(p));
                float d = dx.x+dx.y+dy.x+dy.y;
                if (isnan(d))
                  d = 1;
                float a = v.a * pow(min(1, d*512/8), 2.0);
                col = col*(1-a) + v*a;
                
              }
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
