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
        size = V(1024, 768)
        ZglAppWX.__init__(self, size = size, viewControl = OrthoCamera())

        vortexFP = CGShader('fp40', '''
          #include "perlin.cg"

          uniform float2 gridSize;
          uniform float time;

          float2 rot90(float2 v)
          {
            return float2(-v.y, v.x);
          }

          float3 f(float3 p)
          {
            float3 res = gnoise3d(p);
            res += gnoise3d(p*float3(1.9, 1.9, 1.0));
            return res;
          }

          float4 main(float2 p : TEXCOORD0) : COLOR
          {
            p.x *= gridSize.x / gridSize.y;
            float2 v ;
            p*= 5;
            float t = 1.0* time;
            v.x = noise3d(float3(p, t));
            v.y = noise3d(float3(p, t+9.5));

            v *= 3.0;

            return float4(v, 0, 0);
          }
        ''')
        vortexFP.gridSize = size
        setup_perlin(vortexFP)
        flowBuf = RenderTexture( size = size / 2, format=GL_RGBA_FLOAT16_ATI)
        flowBuf.tex.filterLinear()
        @with_(glprofile('updateFlow'))
        def updateFlow():
            with ctx(flowBuf, ortho, vortexFP(time = self.time)): # glstate(GL_BLEND)
                drawQuad()
                
                

        noiseTex = Texture2D(random.rand(size[1], size[0], 4).astype(float32))
        noiseTex.genMipmaps()
        noiseTex.filterLinearMipmap()
        noiseTex.aniso = 4
        
        flowVisFP = CGShader('fp40', '''
          #line 62
          uniform sampler2D flow;
          uniform sampler2D src;
          uniform sampler2D noise;
          uniform float2 gridSize;
          uniform float time;

          const float pi = 3.141593;

          float getnoise(float2 p, float2 vel)
          {
            float2 nvel = normalize(float2(-vel.y, vel.x));
            float4 rnd = tex2D(noise, p/*, vel/gridSize, nvel/gridSize*/);
            //float v = rnd.r;//frac(rnd.r + time);
            float v = sin(2*pi*rnd.r + time * 10.0)*0.5 + 0.5;
            return v;
          }

          float4 main(float2 p : TEXCOORD0) : COLOR
          {
            float2 vel = tex2D(flow, p).xy;
            float2 sp = p * gridSize - vel;
            
            float a = tex2D(src, sp / gridSize).r;
            float b = getnoise(p, vel);
            float v = lerp(a, b, 0.05);//0.05
            float speed = length(vel);
            return float4(v, speed, 0, 0);
          }
        ''')
        flowVisFP.noise = noiseTex
        flowVisFP.flow = flowBuf.tex
        flowVisFP.gridSize = size
        visBuf = PingPong(size = size, format = GL_RGBA_FLOAT16_ATI)
        visBuf.texparams(*Texture2D.Linear)
        with visBuf.src:
            clearGLBuffers()

        
        @with_(glprofile('updateFlowVis'))
        def updateFlowVis():
            with ctx(visBuf.dst, ortho, flowVisFP(src = visBuf.src.tex, time = self.time)):
                drawQuad()
            visBuf.flip()

        texlookupFP = genericFP('''
          #line 104
          float4 data = tex2D(s_texture, tc0.xy);
          float v = data.r;
          float speed = data.g;

          float4 c1 = float4(0.5, 0.5, 0.5, 1.0);
          float4 c2 = float4(v, v, v, 1.0);
          float4 c = lerp(c1, c2, 2*speed);

          return c2;
        ''' )
        def display():
            updateFlow()
            updateFlowVis()
            with ctx(self.viewControl.vp, ortho, texlookupFP( s_texture = visBuf.src.tex )):
                drawQuad()

        self.display = display




if __name__ == "__main__":
    App().run()
