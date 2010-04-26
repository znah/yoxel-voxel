#from __future__ import with_statement
from zgl import *
    
class App(ZglAppWX):
    RayNum  = Int(13)
    RayMult = Int(4)
    SegmentWidth = Float(0.2)


    def __init__(self):
        ZglAppWX.__init__(self, viewControl = OrthoCamera())

        fragProg = genericFP('''
          float2 pos = tc0;
          float d = length(pos) * (sin(f_time*5)*0.1 + 1);
          float a = atan2(pos.y, pos.x) / (2*pi) + 0.5 + f_time*0.1 + sin(50*(d+0.5*f_time))*0.002;


          
          float seg = floor(d*10);
          a -= seg/(13.0*4.0);
          a = frac(a*13);
          
          float c = 0;     
          if (a*4.0 < 1.0 || d < 0.03)
            c = 1;

          

          return float4(c, c, c, 1);
        
        ''')
    
        def display():
            clearGLBuffers()
            with ctx(self.viewControl.with_vp, fragProg(f_time = self.time)):
                drawQuad(rect = self.viewControl.rect)
        self.display = display

if __name__ == "__main__":
    App().run()
