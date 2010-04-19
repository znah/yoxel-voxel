from __future__ import with_statement
from zgl import *
    
class App(ZglAppWX):
    threshold = Range(0.0, 1.0, 0.5)
    blur = Range(0.0, 0.2, 0.05)

    def __init__(self):
        ZglAppWX.__init__(self, viewControl = OrthoCamera())
        fragProg = genericFP(''' 
          float2 p = tc0.xy;
          float2 dp = p - f2_mpos;
          float d = length(dp);

          float v = tex2D(s_texture, lerp(p, f2_mpos, saturate(0.5-d))) + sin(saturate(0.2 - d*0.5)*100)*0.1; 
          return saturate( (v - f_threshold) / f_blur);
        ''')
        fragProg.s_texture = loadTex('data/highmap.jpg')

    
        def display():
            clearGLBuffers()
            fragProg( 
              f_threshold = self.threshold, 
              f_blur = self.blur, 
              f2_mpos = self.viewControl.mPosWld )

            with ctx(self.viewControl.with_vp, fragProg):
                drawQuad(rect = self.viewControl.rect)
        self.display = display

if __name__ == "__main__":
    App().run()
