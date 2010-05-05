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
        size = V(1600, 1024)
        ZglAppWX.__init__(self, size = size, viewControl = OrthoCamera())

        vortexFP = CGShader('fp40', '''
          #include "perlin.cg"

          uniform float2 gridSize;
          uniform float time;

          float2 rot90(float2 v)
          {
            return float2(-v.y, v.x);
          }

          float2 g(float3 p)
          {
            float h = 0.1;
            float v0 = noise3d(p);
            float dx = noise3d(p+float3(h, 0, 0))-v0;
            float dy = noise3d(p+float3(0, h, 0))-v0;
            return float2(dx, dy) / h;
          }


          float4 main(float2 p : TEXCOORD0) : COLOR
          {
            p.x *= gridSize.x / gridSize.y;
            float t = 1.0*time;
            float2 v = float2(0);
            p *= 2.0;
            
            float a = 1.0;
            float s = 1.0;
            for (int i = 0; i < 4; ++i)
            {
              v += a*g(float3(p*s, t*(i+1)*0.9123));
              s *= 2.9123132;
              a *= 0.6;
            }
            v += 2.0*rot90(v);
            v *= 2.0;

            return float4(v, 0, 0);
          }
        ''')
        vortexFP.gridSize = size
        setup_perlin(vortexFP)
        flowBuf = RenderTexture( size = size / 4, format=GL_RGBA_FLOAT16_ATI)
        flowBuf.tex.filterLinear()
        @with_(glprofile('updateFlow'))
        def updateFlow():
            with ctx(flowBuf, ortho, vortexFP(time = self.time)): # glstate(GL_BLEND)
                drawQuad()
                
                

        noiseTex = Texture2D(random.rand(size[1], size[0], 4).astype(float32))
        noiseTex.genMipmaps()
        #noiseTex.filterLinearMipmap()
        noiseTex.setParams(*Texture.Repeat)
        
        flowVisFP = CGShader('fp40', '''
          #line 83
          uniform sampler2D flow;
          uniform sampler2D src;
          uniform sampler2D noise;
          uniform float2 gridSize;
          uniform float time;

          const float pi = 3.141593;

          float getnoise(float2 p)
          {
            float4 rnd = tex2D(noise, p);
            float v = sin(2*pi*rnd.r + time * 10.0)*0.5 + 0.5;
            v *= saturate(sin(p.x*20));
            return v;
          }
          
          float getnoise_lerp(float2 p)
          {
            float2 dp = 1.0f / gridSize;
            p *= gridSize;
            float2 f = frac(p);
            p = dp * (p-f);

            float v0 = lerp(getnoise(p), getnoise(p + float2(dp.x, 0)), f.x);
            p.y += dp.y;
            float v1 = lerp(getnoise(p), getnoise(p + float2(dp.x, 0)), f.x);
            return lerp(v0, v1, f.y);
          }

          float4 main(float2 p : TEXCOORD0) : COLOR
          {
            float2 vel = tex2D(flow, p).xy;
            float2 sp = p - vel / gridSize;
            float speed = length(vel);

            float3 a = float3(0);
            float3 b = float3(0);
            float c = 0;
            float dt = 1.0 / speed;
            for (float t = 0; t < 1.0; t += dt)
            {
              float2 pm = lerp(sp, p, t);
              a += tex2D(src, pm).rgb;
              b += float3(getnoise(pm));
              
              b.gb *= saturate(speed*0.5);

              c += 1.0;
            }
            a /= c;
            b /= c;

            float3 v = lerp(a, b, 0.05);
            return float4(v, speed);
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
          float3 v = data.rgb;
          float speed = data.a;

          v = lerp(float3(0.5), v, 2.0);

          return float4(v, 1.0);
        ''' )
        self.lastTime = 0
        dt = 1.0/60.0

        def display():
            if self.time - self.lastTime > dt:
                updateFlow()
                updateFlowVis()
                self.lastTime = self.time

            with ctx(self.viewControl.vp, ortho, texlookupFP( s_texture = visBuf.src.tex )):
                drawQuad()

        self.display = display




if __name__ == "__main__":
    App().run()
