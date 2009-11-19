from __future__ import with_statement
from zgl import *

class ReactionDiffusion(object):
    def __init__(self, size = (256, 256)):
        self.size = size

        self.reactFrag = CGShader("fp40", """
          uniform sampler2D tex;
          uniform float2 dpos;
          uniform float2 f_k;
          uniform float dt;
          float4 main(float2 pos : TEXCOORD0) : COLOR
          {
            float2 v = tex2D(tex, pos).xy;
            float2 l = -4.0 * v;
            l += tex2D(tex, pos + float2( dpos.x, 0)).xy;
            l += tex2D(tex, pos + float2(-dpos.x, 0)).xy;
            l += tex2D(tex, pos + float2( 0, dpos.y)).xy;
            l += tex2D(tex, pos + float2( 0,-dpos.y)).xy;

            const float2 diffCoef = float2(0.082, 0.041);

            const float f = f_k.x;
            const float k = f_k.y;

            float2 dv = diffCoef * l;
            float rate = v.x * v.y * v.y;
            dv += float2(-rate, rate);
            dv += float2(f * (1.0 - v.x), -(f + k) * v.y );
            v += dt * dv;
            return float4(v, rate, 0);
          }
        """)

        a = zeros(YX(size) + (4,), float32)
        a[...,0] = 1
        self.pp = PingPong(img = a, format = GL_RGBA_FLOAT16_ATI)
        self.reactFrag.dpos = 1.0 / V(size)

        self.reactFrag.f_k = (0.029, 0.055)
        self.reactFrag.dt = 0.5

    def iterate(self, n = 1):
        self.reactFrag.tex = self.pp.src.tex
        with ctx(self.pp.dst, self.reactFrag, ortho):
            drawQuad()
        self.pp.flip()

    def state(self):
        return self.pp.src
        

class App(ZglApp):
    def __init__(self):
        ZglApp.__init__(self, OrthoCamera())

        self.rd = ReactionDiffusion( (256, 256) )

        self.fragProg = CGShader('fp40', '''
          uniform sampler2D tex;
          float4 main( float2 tc: TEXCOORD0 ) : COLOR 
          { 
            return tex2D(tex, tc); 
          }
        ''')
    
    def display(self):
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self.rd.iterate(5)
        self.fragProg.tex = self.rd.state().tex
        with ctx(self.viewControl.with_vp, self.fragProg):
            drawQuad()

        glutSwapBuffers()

    def dropDrop(self, x, y):
        with ctx(self.rd.state(), ortho):
            glColor4f(0.5, 0.25, 0, 0)
            glPointSize(10.0)
            glBegin(GL_POINTS)
            glVertex(x, y)
            glEnd()

    def mouseButton(self, btn, up, x, y):
        self.dropDrop(*self.viewControl.mPosWld)
        ZglApp.mouseButton(self, btn, up, x, y)


if __name__ == "__main__":
  viewSize = (800, 600)
  zglInit(viewSize, "hello")

  glutSetCallbacks(App())

  #wglSwapIntervalEXT(0)
  glutMainLoop()
