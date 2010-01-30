from __future__ import with_statement
from zgl import *
    
class App(ZglAppWX):
    dcoef = Range(0.0, 100.0, 10.0)
    iterNum = Range(1, 100, 30)
    
    @on_trait_change( 'iterNum' )
    def buildShader(self):
        self.fragProg = CGShader('fp40', '''
          #line 13
          uniform float time;
          uniform float dcoef;

          float2 cmul(float2 a, float2 b)
          {
            float2 c;
            c.x = a.x*b.x - a.y*b.y;
            c.y = a.x*b.y + a.y*b.x;
            return c;
          }
          
          float4 main( float2 c: TEXCOORD0 ) : COLOR 
          { 
            float2 z = c;
            const int N = %(iterNum)d;
            for (int i = 0; i < N; ++i)
            {
              z = cmul(z, z) + c;
              if (z.x*z.x + z.y*z.y > 4)
              {
                float v = float(i) / N;
                return float4(v, v, v, 1);
              }
            }

            float d = length(z);
            float v = abs(sin(d*dcoef + time));
            return float4(v, v, v, 1);
          }

        ''' % {'iterNum' : self.iterNum})
    
    def __init__(self):
        ZglAppWX.__init__(self, viewControl = OrthoCamera())
        self.buildShader()
    
    def display(self):
        clearGLBuffers()
        self.fragProg.dcoef = self.dcoef
        with ctx(self.viewControl.with_vp, self.fragProg(time = self.time)):
            drawQuad(self.viewControl.rect)

if __name__ == "__main__":
    App().run()
