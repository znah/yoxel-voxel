from __future__ import with_statement
from zgl import *
from PIL import Image
from time import clock

class App(ZglApp):
    def __init__(self):
        ZglApp.__init__(self, OrthoCamera())

        self.viewControl.rect = (-2, -1, 2, 1)

        def loadTex(fn):
            tex = Texture2D(Image.open(fn))
            tex.filterLinearMipmap()
            tex.genMipmaps()
            tex.setParams( (GL_TEXTURE_MAX_ANISOTROPY_EXT, 16))
            return tex

        self.tex1 = loadTex("img/fung.png")
        self.tex2 = loadTex("img/lines.png")
        self.tex3 = loadTex("img/bubble.png")
        self.noiseTex = loadTex("img/noise256g.png")


        self.fragProg = CGShader('fp40', '''
          uniform sampler2D tex;
          uniform sampler2D noiseTex;
          uniform float time;
          uniform float2 juliaSeed;
                     
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
            
            //float2 c = tc;
            //float2 z = float2(0, 0);
            
            float2 c = juliaSeed;
            float2 z = tc;

            float os = 0.2 * time;

            for (int i = 0; i < 30; ++i)
            {
              //float2 z2 = cmul(z, z);
              z = cmul(z, z) + c;
              {
                float2 p = z;
                //p = up*p.x + vp*p.y;

                float ox = 2*tex2D(noiseTex, (p+os)*0.11 + 0.3).r-1;
                float oy = 2*tex2D(noiseTex, (p+os)*0.11 + 0.8).r-1;
                float2 ofs = float2(ox, oy);
                p += ofs * 0.03;

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
        self.fragProg.tex = self.tex1
        self.fragProg.noiseTex = self.noiseTex

        self.juliaSeed = V(-0.726895347709114071439, 0.188887129043845954792)
        self.fragProg.juliaSeed = self.juliaSeed

        self.shotn = 0

    
    def display(self):
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self.fragProg.time = clock()
        with ctx(self.viewControl.with_vp, self.fragProg):
            drawQuad(self.viewControl.rect)

        glutSwapBuffers()

    def mouseMove(self, x, y):
        if self.viewControl.mButtons[2]:
           dp = self.viewControl.scr2wld(x, y) - self.viewControl.mPosWld
           self.juliaSeed += dp
           self.fragProg.juliaSeed = self.juliaSeed
        ZglApp.mouseMove(self, x, y)

    def keyDown(self, key, x, y):
        if key == '1':
            self.fragProg.tex = self.tex1
        if key == '2':
            self.fragProg.tex = self.tex2
        if key == '3':
            self.fragProg.tex = self.tex3
        if key == 'w':
            self.screenshot()
        ZglApp.keyDown(self, key, x, y)

    def screenshot(self):
        sz = self.viewControl.vp.size
        pixels = glReadPixels(0, 0, sz[0], sz[1], GL_RGB, GL_UNSIGNED_BYTE, 'array')
        pixels.shape = (sz[1], sz[0], 3) # !!! bug
        pixels = flipud(pixels)
        img = Image.fromarray(pixels)
        img.save("shot_%02d.png" % (self.shotn,))
        self.shotn += 1



if __name__ == "__main__":
  viewSize = (800, 600)
  zglInit(viewSize, "hello")

  glutSetCallbacks(App())

  #wglSwapIntervalEXT(0)
  glutMainLoop()
