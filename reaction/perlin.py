from __future__ import with_statement
from zgl import *

def setup_perlin(prog):
    perm = random.permutation(256).astype(uint8)
    prog.uPerlinPerm = Texture1D(img = perm[:,newaxis])

    perm2d = zeros((256, 256, 4), uint8)
    x = arange(256)
    y = x.reshape(-1, 1)
    a = (perm[x] + y)%256
    b = (perm[(x+1)%256] + y)%256
    aa = perm[a]
    ab = perm[(a+1)%256]
    ba = perm[b]
    bb = perm[(b+1)%256]
    perm2d = dstack((aa, ba, ab, bb)).astype(uint8)
    '''
      float A  = _perm(p0.x) + p0.y;
      float AA = _perm(A);
      float AB = _perm(A + one);
      float B  = _perm(p0.x + one) + p0.y;
      float BA = _perm(B);
      float BB = _perm(B + one);
      return float4(AA, BA, AB, BB);
    '''
    prog.uPerlinPerm2d = Texture2D(img = perm2d)
    

    grad = array([
        1,1,0,    -1,1,0,    1,-1,0,    -1,-1,0,
        1,0,1,    -1,0,1,    1,0,-1,    -1,0,-1,
        0,1,1,    0,-1,1,    0,1,-1,    0,-1,-1,
        1,1,0,    0,-1,1,    -1,1,0,    0,-1,-1], float32).reshape(-1,3)
    gradperm = grad[perm%16]
    prog.uPerlinGrad = Texture1D(img = grad, format = GL_RGBA_FLOAT16_ATI)
    prog.uPerlinGradPerm = Texture1D(img = gradperm, format = GL_RGBA_FLOAT16_ATI)
    

if __name__ == "__main__":
    class App(ZglAppWX):
        def __init__(self):
            ZglAppWX.__init__(self, viewControl = OrthoCamera(), size = (1024, 768))
            self.fragProg = CGShader('fp40', '''
             #include "perlin.cg"

             uniform float time;

             float4 main(float2 pos: TEXCOORD0) : COLOR
             {
               return float4(noise3d(pos.x*50, pos.y*50, time));
    
             }

            ''')
            setup_perlin(self.fragProg)
        
        def display(self):
            clearGLBuffers()
            with ctx(self.viewControl.with_vp, self.fragProg(time = self.time)):
                with glprofile('perlin'):
                    drawQuad()
    
    App().run()
