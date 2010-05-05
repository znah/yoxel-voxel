from __future__ import with_statement
from zgl import *

def setup_perlin(prog):
    perm = random.permutation(256).astype(uint8)[:,newaxis]
    permTex = Texture1D(img = perm)
    prog.uPerlinPerm = permTex

    grad = array([
        1,1,0,    -1,1,0,    1,-1,0,    -1,-1,0,
        1,0,1,    -1,0,1,    1,0,-1,    -1,0,-1,
        0,1,1,    0,-1,1,    0,1,-1,    0,-1,-1,
        1,1,0,    0,-1,1,    -1,1,0,    0,-1,-1], float32).reshape(-1,3)
    prog.uPerlinGrad = Texture1D(img = grad, format = GL_RGBA_FLOAT16_ATI)

    
class App(ZglAppWX):
    def __init__(self):
        ZglAppWX.__init__(self, viewControl = OrthoCamera())
        self.fragProg = CGShader('fp40', '''
         #include "perlin.cg"

         float4 main(float2 pos: TEXCOORD0) : COLOR
         {
          return float4(gnoise3d(float3(pos*10, 1.0)), 1);

         }

        ''')
        setup_perlin(self.fragProg)
    
    def display(self):
        clearGLBuffers()
        with ctx(self.viewControl.with_vp, self.fragProg):
            drawQuad()

if __name__ == "__main__":
    App().run()
