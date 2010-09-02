from __future__ import with_statement
from zgl import *
from perlin import setup_perlin

    
class App(ZglAppWX):

    gamma      = Range(0.0, 5.0, 1.0)
    convolStep = Range(0.5, 5.0, 2.0)
    blendCoef  = Range(0.0, 1.0, 0.1)
    
    
    def __init__(self):
        size = V(900, 600)
        ZglAppWX.__init__(self, size = size, viewControl = OrthoCamera())

        vortexFP = CGShader('fp40', '''
          #line 19
          #include "perlin.cg"

          uniform float2 gridSize;
          uniform float time;

          float2 rot90(float2 v)
          {
            return float2(-v.y, v.x);
          }

          float2 g(float3 p)
          {
            /*float h = 0.1;
            float v0 = noise3d(p);
            float dx = noise3d(p+float3(h, 0, 0))-v0;
            float dy = noise3d(p+float3(0, h, 0))-v0;
            return float2(dx, dy) / h;*/
            return snoiseGrad(p*0.8).xy*0.4;

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

            //float2 c = float2(1, 0.8) * gridSize.x / gridSize.y;
            //float d = length(c-p)*1.0;
            //v = lerp((p-c)*20, v, saturate(d));


            return float4(v, 0, 0);
          }
        ''')
        vortexFP.gridSize = size
        setup_perlin(vortexFP)
        flowBuf = RenderTexture( size = size / 4, format=GL_RGBA_FLOAT16_ATI)
        flowBuf.tex.filterLinear()
        @with_(glprofile('updateFlow'))
        def updateFlow():
            with ctx(flowBuf, ortho01, vortexFP(time = self.time)): # glstate(GL_BLEND)
                drawQuad()
                
                

        noiseTex = Texture2D(random.rand(size[1], size[0], 4).astype(float32))
        
        flowVisFP = CGShader('fp40', '''
          #line 82
          uniform sampler2D flow;
          uniform sampler2D src;
          uniform sampler2D noise;
          uniform float2 gridSize;
          uniform float time;
          uniform float2 mousePos;
          uniform float convolStep;
          uniform float blendCoef;

          const float pi = 3.141593;

          float getnoise(float2 p)
          {
            float4 rnd = tex2D(noise, p);
            //float v = rnd.r;
            float v = sin(2*pi*rnd.r + time * 0.1)*0.5 + 0.5;
            //if (v < 0.99)
            //  v = 0;
            //v *= 20;
            //v *= saturate(sin(p.x*10));
            return v;
          }
          
          float4 main(float2 p : TEXCOORD0) : COLOR
          {
            float2 vel = tex2D(flow, p).xy;
            float2 sp = p - vel / gridSize;
            float speed = length(vel);

            float3 a = float3(0);
            float3 b = 0;
            float noiseAccum = 0;
            float c = 0;
            float dt = convolStep / speed;

            float2 gp = p * gridSize;
            float dist = length(mousePos - gp);
            
            //float3 col = lerp(float3(2.5, 1.5, 0), 
            //                  float3(1, 0, 0), 
            //                  saturate(0.1*speed));
            float3 col = lerp(float3(1, 0, 0), 
                              float3(1, 1, 1), 
                              saturate(0.3*speed));

            for (float t = 0; t < 1.0; t += dt)
            {
              float2 pm = lerp(sp, p, t);
              a += tex2D(src, pm).rgb;
              noiseAccum += getnoise(pm);

              c += 1.0;
            }
            a /= c;
            noiseAccum /= c;
            b = noiseAccum * col;

            float3 v = lerp(a, b, blendCoef);
            return float4(v, speed);
            //return float4(vel, 0, 1);
          }
        ''')
        flowVisFP.noise = noiseTex
        flowVisFP.flow = flowBuf.tex
        flowVisFP.gridSize = size

        visBuf = PingPong(size = size)#, format = GL_RGBA_FLOAT16_ATI)
        visBuf.texparams(*Texture2D.Linear)
        with visBuf.src:
            clearGLBuffers()

        
        @with_(glprofile('updateFlowVis'))
        def updateFlowVis():
            x, y = self.viewControl.mPos
            y = self.viewControl.vp.size[1]-y-1
            flowVisFP(mousePos = (x, y),
                      convolStep = self.convolStep,
                      blendCoef = self.blendCoef)
            with ctx(visBuf.dst, ortho01, flowVisFP(src = visBuf.src.tex, time = self.time)):
                drawQuad()
            visBuf.flip()

        texlookupFP = genericFP('''
          #line 104
          float4 data = tex2D(s_texture, tc0.xy);
          float3 v = data.rgb;
          float speed = data.a;

          //v = lerp(float3(0.5), v, 1.0+f_contrast);
          v = pow(v, f_gamma);

          return float4(v, 1.0);
        ''' )
        self.lastTime = 0
        dt = 1.0/50.0

        def display():
            if self.time - self.lastTime > dt:
                with profile('update'):
                    updateFlow()
                    updateFlowVis()
                self.lastTime = self.time

            texlookupFP(
              s_texture = visBuf.src.tex,
              f_gamma = self.gamma)
            with ctx(self.viewControl.vp, ortho01, texlookupFP( s_texture = visBuf.src.tex )):
                drawQuad()

        self.display = display




if __name__ == "__main__":
    App().run()
