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

    gradperm2d = zeros((256, 256, 4), uint8)
    gradperm2d[:,:,3] = perm2d[:,:,0]
    gradperm2d[:,:,0:3] = grad[perm2d[:,:,0]%16] * 127 + 127

    
    
    a = random.rand(256, 256, 4)
    g = a[...,:3]
    g[:] = 2.0*g-1.0
    g /= (sum(g*g, axis=2)**0.5)[...,newaxis]
    g *= sqrt(2.0)
    g[:] = 0.5*g+0.5
    gradperm2d = (a*255).astype(uint8)
    

    im = PIL.Image.fromarray(gradperm2d)
    im.save('t.png')

    prog.uPerlinGradPerm2d = Texture2D(img = gradperm2d)


    

if __name__ == "__main__":
    class App(ZglAppWX):
        def __init__(self):
            ZglAppWX.__init__(self, viewControl = OrthoCamera(), size = (1024, 768))
            self.fragProg = CGShader('fp40', '''
             #include "perlin.cg"

             uniform float time;


             float4 main(float2 pos: TEXCOORD0) : COLOR
             {
               float3 p = float3(pos*50, 0);
               //float c = 0.7071;
               //p = float3((p.x-p.z)*c, p.y, (p.x+p.z)*c);
               //return 0.5*noise3d(p)+0.5;

               float4 v = snoiseGrad(p);
               return length(v.xy)*0.2;
               
               /*float ac = 0.5;//0.5*snoise(float3(pos.x*10, pos.y*10, 23.0))+0.5;

               float a = 1.0, s = 5.0, v = 0;
               for (int i = 0; i < 8; ++i)
               {
                 v += a * snoise2(float3(pos.x*s, pos.y*s, time+i*0.123));
                 a *= ac;
                 s *= 1.96876;
               }
               return float4(abs(v));*/
             }

            ''')
            setup_perlin(self.fragProg)
        
        def display(self):
            clearGLBuffers()
            with ctx(self.viewControl.with_vp, self.fragProg(time = self.time)):
                with glprofile('perlin'):
                    drawQuad()
    
    App().run()
