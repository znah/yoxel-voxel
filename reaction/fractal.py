from __future__ import with_statement
from zgl import *

class App(ZglAppWX):
    def __init__(self):
        ZglAppWX.__init__(self, size = (960, 600), viewControl = OrthoCamera())

        self.viewControl.rect = (-2, -1, 2, 1)

        self.textFrag = CGShader('fp40', '''
          uniform sampler2D fontTex;
          uniform float2 dp;
          float4 main(float2 pos : TEXCOORD0) : COLOR
          {
            float c = tex2D(fontTex, pos);
            return float4(0, c, 0, c);
          }
        ''')
        font = asarray(PIL.Image.open("img/font.png").convert('L')).copy()
        self.textFrag.fontTex = fontTex = Texture2D(font)
        self.textFrag.dp = 1.0 / fontTex.size

        self.tex = {}
        self.tex['1'] = loadTex("img/fung.png")
        self.tex['2'] = loadTex("img/lines.png")
        self.tex['3'] = loadTex("img/bubble.png")
        self.tex['4'] = loadTex("img/tentacle.png")

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

            const float Escape = 10.0;

            for (int i = 0; i < 30; ++i)
            {
              z = cmul(z, z) + c;
              if ( any( abs(z) > Escape ) )
                break;
              
              {
                float2 p = z;

                float2 ofs = 2*tex2D(noiseTex, (p+os)*0.1).rg-1;
                p += ofs * 0.035;

                float fade = min(1, 2/(abs(p.x)+abs(p.y)));
                p += float2(0.5, 0.5) + float2(i*0.34, i*0.12);
                float4 v = tex2D(tex, p) * fade;
                
                float2 dx = abs(ddx(p));
                float2 dy = abs(ddy(p));
                float d = dx.x+dx.y+dy.x+dy.y;
                float a = v.a * pow(min(1, d*512/8), 2.0);
                col = col*(1-a) + v*a;
              }
            }
            return float4(col);
          }
        ''')
        self.fragProg.tex = self.tex['1']
        self.fragProg.noiseTex = loadTex("img/noise256x4g.png")

        self.juliaSeed = V(-0.726895347709114071439, 0.188887129043845954792)
        self.fragProg.juliaSeed = self.juliaSeed

        self.largeBuf = RenderTexture(size=(1280, 800))

        self.shotn = 0

    def drawTextLine(self, pos, s):
        w, h = 8, 16
        ch = fromstring(s, uint8)
        n = len(s)
        cx = (ch % 16) * w
        cy = (ch / 16) * h
        px = arange(n)*w + pos[0]
        py = pos[1]
        
        v = array([[0, 0], [w, 0], [w, h], [0, h]], float32)
        pos = tile(v, (n, 1, 1))
        pos[...,0] += px[:,newaxis]
        pos[...,1] += py
        tc = tile(v, (n, 1, 1))
        tc[..., 0] += cx[:,newaxis]
        tc[..., 1] += cy[:,newaxis]
        drawArrays(GL_QUADS, verts = pos, tc0 = tc/(128, 256))

    @with_(glprofile('drawText_gl'))
    def drawText(self, pos, s):
        x, y = pos
        for s in s.splitlines():
            self.drawTextLine((x, y), s)
            y += 16
        
    @with_(glprofile('frame_gl'))
    def display(self):
        clearGLBuffers()
        self.fragProg.time = clock()
        with self.viewControl.with_vp:
            with ctx(glprofile('fractal_gl'), self.fragProg):
                drawQuad(self.viewControl.rect)
            sx, sy = self.viewControl.vp.size
            rect = (0, sy, sx, 0)
            with ctx( self.textFrag, glstate(GL_BLEND), Ortho(rect) ):
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                glBlendEquation(GL_FUNC_ADD);
                with profile("dumpProfile"):
                    s = dumpProfile()
                self.drawText((50, 100), s)

    def OnMouse(self, evt):
        if self.viewControl.mButtons[2]:
            x, y = evt.Position
            dp = self.viewControl.scr2wld(x, y) - self.viewControl.mPosWld
            self.juliaSeed += dp
            self.fragProg.juliaSeed = self.juliaSeed
        ZglAppWX.OnMouse(self, evt)
        

    def OnKeyDown(self, evt):
        code = evt.GetKeyCode()
        if code < 256:
            key = chr(code)
            if self.tex.has_key(key):
                self.fragProg.tex = self.tex[key]
            if key == 'W':
                self.screenshot()
            if key == 'S':
                self.renderShot()
            if key == ' ':
                print dumpProfile()
        ZglAppWX.OnKeyDown(self, evt)

    def screenshot(self):
        sz = self.viewControl.vp.size
        pixels = glReadPixels(0, 0, sz[0], sz[1], GL_RGB, GL_UNSIGNED_BYTE, 'array')
        pixels.shape = (sz[1], sz[0], 3) # !!! bug
        pixels = flipud(pixels)
        img = PIL.Image.fromarray(pixels)
        img.save("shot_%02d.jpg" % (self.shotn,))
        self.shotn += 1
    
    def renderShot(self):
        with glprofile("renderShot"):
            sz = self.largeBuf.size()
            #self.fragProg.time = clock()
            with ctx(self.largeBuf, self.fragProg, Ortho(self.viewControl.rect)):
                drawQuad(self.viewControl.rect)
                pixels = glReadPixels(0, 0, sz[0], sz[1], GL_RGB, GL_UNSIGNED_BYTE, 'array')
            pixels.shape = (sz[1], sz[0], 3) # !!! bug
            pixels = flipud(pixels)
            img = PIL.Image.fromarray(pixels)
            img.save("shot_%02d.jpg" % (self.shotn,))
            self.shotn += 1
    

if __name__ == "__main__":
    App().run()
