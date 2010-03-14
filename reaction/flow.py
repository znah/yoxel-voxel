from __future__ import with_statement
from zgl import *
    
class App(ZglAppWX):
    
    
    def __init__(self):
        size = V(1920, 1000)
        ZglAppWX.__init__(self, size = size, viewControl = OrthoCamera())

        vortexFP = CGShader('fp40', '''
          uniform float2 gridSize;
          uniform float2 center;
          
          uniform float turn;
          uniform float emit;
          uniform float fade;

          float4 main(float2 p : TEXCOORD0) : COLOR
          {
            float2 r = p * gridSize - center;
            float d = length(r);
            r /= d;
            d /= gridSize.x;
            float2 t = float2(-r.y, r.x);
            
            t *= turn*d * exp(-d*d*fade);
            return float4(t + emit*r/d, 0, 0);
          }
        ''')
        vortexFP.gridSize = size
        flowBuf = RenderTexture( size = size / 2, format=GL_RGBA_FLOAT16_ATI)
        flowBuf.tex.filterLinear()
        @with_(glprofile('updateFlow'))
        def updateFlow():
            with ctx(flowBuf, ortho, vortexFP , glstate(GL_BLEND)):
                glBlendFunc(GL_ONE, GL_ONE);
                glBlendEquation(GL_FUNC_ADD);
                clearGLBuffers()
                vortexFP(turn = 30.0, emit = 0.0, fade = 30, center = size*(0.3, 0.3))
                drawQuad()

                if self.viewControl.mButtons[0]:
                    emit = 0.005
                    turn = 30.0
                elif self.viewControl.mButtons[2]:
                    emit = -0.005
                    turn = -30.0
                else:
                    emit = 0
                    turn = -30.0
                vortexFP(turn = turn, emit = emit)
                pos = self.viewControl.mPos
                pos = (pos[0], size[1]-pos[1])
                vortexFP(center = pos)
                drawQuad()

        noiseTex = Texture2D(random.rand(size[1], size[0], 4).astype(float32))
        #noiseTex.filterLinear()
        flowVisFP = CGShader('fp40', '''
          uniform sampler2D flow;
          uniform sampler2D src;
          uniform sampler2D noise;
          uniform float2 gridSize;
          uniform float time;

          const float pi = 3.141593;

          float getnoise(float2 p)
          {
            float4 rnd = tex2D(noise, p);
            float v = frac(rnd.r + time);//sin(2*pi*rnd.r + time * 10.0)*0.5 + 0.5;
            return pow(v, 4.0);
          }

          float4 main(float2 p : TEXCOORD0) : COLOR
          {
            float2 vel = tex2D(flow, p).xy;
            float2 sp = p * gridSize - vel;
            
            float a = tex2D(src, sp / gridSize).r;
            float b = getnoise(p);

            b = lerp(0.0, b, length(vel)*1);
            return float4(float3(lerp(a, b, 0.05)), 1.0);
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
            with ctx(visBuf.dst, ortho, flowVisFP(src = visBuf.src.tex, time = self.time)):
                drawQuad()
            visBuf.flip()

        texlookupFP = genericFP('tex2D(texture, tc0.xy)')
        def display():
            updateFlow()
            updateFlowVis()
            with ctx(self.viewControl.vp, ortho, texlookupFP( texture = visBuf.src.tex )):
                drawQuad()

        self.display = display




if __name__ == "__main__":
    App().run()
