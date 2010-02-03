from __future__ import with_statement
from zgl import *


class EmptyFlowVis(object):
    def __init__(self):
        pass

    def reset(self):
        pass

    def update(self, dt, time, flowTex):
        pass

    def render(self):
        pass


class TexFlowVis(object):
    def __init__(self):
        x = linspace(0, 1, 1024)
        X, Y = meshgrid(x, x)
        a = zeros((len(x), len(x), 4), float32)
        a[...,0] = X
        a[...,1] = Y
        self.sandTex = PingPong(img=a, format = GL_RGBA_FLOAT32_ATI)
        self.sandTex.texparams(*Texture2D.Linear)
        self.sandUpdate = CGShader('fp40', '''
          uniform sampler2D flowTex;
          uniform sampler2D sandTex;
          uniform float dt;
          float4 main( float2 tc: TEXCOORD0 ) : COLOR
          {
            float2 v = tex2D(flowTex, tc).xy;
            float2 p = tc - v*dt;
            return tex2D(sandTex, p);
          }
        ''')

        self.visFrag = CGShader( 'fp40', '''
          uniform sampler2D flowTex;
          uniform sampler2D sandTex;
          uniform sampler2D bgTex;
          uniform float t;
          float4 main( float2 tc: TEXCOORD0 ) : COLOR
          {
            float2 p = tex2D(sandTex, tc).xy;
            return tex2D(bgTex, p);
          }
        ''')
        self.bgTex = loadTex("img\\lush.jpg")
        #a = random.rand(1024, 1024).astype(float32)
        #self.bgTex = Texture2D(img=a)
        #self.bgTex.genMipmaps()
        #self.bgTex.aniso(16)
        #self.bgTex.filterLinearMipmap()
        
        self.visFrag.bgTex = self.bgTex
        self.visFrag.sandTex = self.sandTex.src.tex


    def reset(self):
        pass

    def update(self, dt, time, flowTex):
        self.sandUpdate.sandTex = self.sandTex.src.tex
        self.sandUpdate.dt = dt
        self.sandUpdate.flowTex = flowTex
        with ctx(self.sandTex.dst, self.sandUpdate, ortho):
            drawQuad()
        self.sandTex.flip()

        self.visFrag.flowTex = flowTex
        self.visFrag.sandTex = self.sandTex.src.tex
        self.visFrag.t = time

    def render(self):
        with self.visFrag:
            drawQuad()

class NoiseFlowVis(object):
    def __init__(self):
        a = random.rand(512, 512, 4).astype(float32)
        self.noiseTex = PingPong(img=a, format = GL_RGBA_FLOAT32_ATI)
        self.noiseTex.texparams(*Texture2D.Linear)
        self.sandUpdate = CGShader('fp40', '''
          uniform sampler2D flowTex;
          uniform sampler2D noiseTex;
          uniform sampler2D sandTex;
          uniform float dt;
          float4 main( float2 tc: TEXCOORD0 ) : COLOR
          {
            float2 v = tex2D(flowTex, tc).xy;
            float2 p = tc - v*dt;
            return lerp(tex2D(sandTex, p), tex2D(noiseTex, p)  ) ;
          }
        ''')

        self.visFrag = CGShader( 'fp40', '''
          uniform sampler2D flowTex;
          uniform sampler2D sandTex;
          uniform sampler2D bgTex;
          uniform float t;
          float4 main( float2 tc: TEXCOORD0 ) : COLOR
          {
            float2 p = tex2D(sandTex, tc).xy;
            return tex2D(bgTex, p);
          }
        ''')
        self.bgTex = loadTex("img\\lush.jpg")
        self.visFrag.bgTex = self.bgTex
        self.visFrag.sandTex = self.sandTex.src.tex


    def reset(self):
        pass

    def update(self, dt, time, flowTex):
        self.sandUpdate.sandTex = self.sandTex.src.tex
        self.sandUpdate.dt = dt
        self.sandUpdate.flowTex = flowTex
        with ctx(self.sandTex.dst, self.sandUpdate, ortho):
            drawQuad()
        self.sandTex.flip()

        self.visFrag.flowTex = flowTex
        self.visFrag.sandTex = self.sandTex.src.tex
        self.visFrag.t = time

    def render(self):
        with self.visFrag:
            drawQuad()


        

class App(ZglAppWX):
    def __init__(self):
        ZglAppWX.__init__(self, viewControl = OrthoCamera())

        vortexRowN = 4
        dustRowN = 256
        w = 128
        self.psize = (w, vortexRowN + dustRowN)
        self.vortexOfs = 0
        self.vortexN = vortexRowN * w
        self.dustOfs = self.vortexOfs + self.vortexN
        self.dustN = dustRowN * w

        pstate = random.rand(self.psize[1], self.psize[0], 4).astype(float32)
        pstate[..., 0:2] *= 0.1
        pstate[..., 0:2] += 0.45
        pstate[..., 2] = pstate[..., 2] ** 2
        pstate[..., 2] = 0.02 + (pstate[..., 2])  # radius
        pstate[..., 3] = (2 * pstate[..., 3] - 1)*2 # curl
        self.particles = PingPong(img=pstate, format = GL_RGBA_FLOAT32_ATI)
        self.partVBO = BufferObject(pstate, GL_STREAM_COPY)

        flowBufSize = 256
        self.flowTex = RenderTexture(size = (flowBufSize, flowBufSize), format = GL_RGBA_FLOAT32_ATI)
        self.flowTex.tex.filterLinear()

        particleCode = '''
          uniform sampler2D flowTex;
          uniform sampler2D partTex;
          uniform float t;
          uniform float dt;

          float4 main( float2 tc: TEXCOORD0 ) : COLOR
          {
            float4 data = tex2D(partTex, tc);
            float2 p = data.xy;
            float2 v = tex2D(flowTex, p).xy;
            p += v * dt;
            p = frac(p + float2(1, 1));
            return float4(p, data.z, data.w);
          }
        '''
        self.partUpdateProg = CGShader( 'fp40', particleCode, entry = 'main')

        self.vortexFrag = CGShader('fp40', '''
          float4 main( float2 tc: TEXCOORD0, float curl : TEXCOORD1 ) : COLOR 
          { 
            tc.y = 1.0 - tc.y;
            float2 p = tc * 2.0 - 1.0;
            float2 t = float2(-p.y, p.x) - 0.0*p;
            
            float r = 6*length(p);
            float v = exp(-r) * curl;
            t *= v;

            return float4(t, r*v, 0); 
          }
        ''')
        
        self.vortexVert = CGShader('vp40', '''
          uniform float spriteScale;
          void main( 
            float4 data  : ATTR0,
            out float4 oPos  : POSITION,
            out float oPSize : PSIZE,
            out float oCurl  : TEXCOORD1 ) 
          { 
            float4 p = float4(data.xy, 0, 1);
            oPos = mul(glstate.matrix.mvp, p);
            oPSize = spriteScale * data.z;
            oCurl = data.w;
          }
        ''')
        self.vortexVert.spriteScale = flowBufSize
        
        self.dustVert = CGShader('vp40', '''
          void main( 
            float4 data  : ATTR0,
            out float4 oPos  : POSITION,
            out float4 color : COLOR) 
          { 
            float4 p = float4(data.xy, 0, 1);
            oPos = mul(glstate.matrix.mvp, p);
            color = float4(1);
          }
        ''')


        self.flowVis = TexFlowVis()


        glTexEnvf(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)

        self.paused = True

    def updateParticles(self, dt):
        glBlendFunc(GL_ONE, GL_ONE);
        glBlendEquation(GL_FUNC_ADD);

        flags = [GL_POINT_SPRITE, GL_VERTEX_PROGRAM_POINT_SIZE_ARB, GL_BLEND]
        with ctx(self.flowTex, ortho, self.vortexVert, self.vortexFrag, glstate(*flags)):
            glClearColor(0, 0, 0, 0)
            glClear(GL_COLOR_BUFFER_BIT)
            with self.partVBO.array:
                glVertexAttribPointer(0, 4, GL_FLOAT, False, 0, None)
            with vattr(0):
                glDrawArrays(GL_POINTS, self.vortexOfs, self.vortexN)
            
        self.partUpdateProg.dt = dt
        self.partUpdateProg.partTex = self.particles.src.tex
        self.partUpdateProg.flowTex = self.flowTex.tex
        with ctx(self.particles.dst, ortho, self.partUpdateProg):
            drawQuad()
            with self.partVBO.pixelPack:
                OpenGL.raw.GL.glReadPixels(0, 0, self.psize[0], self.psize[1], GL_RGBA, GL_FLOAT, None)
        self.particles.flip()

        self.flowVis.update(dt, self.time, self.flowTex.tex)

    
    def display(self):
        if self.paused:
            self.simTime = self.time
        else:
            dt = self.time - self.simTime
            tstep = 0.01
            iterNum = int(dt / tstep)
            self.simTime += tstep * iterNum
            if iterNum > 5:
                self.simTime = self.time
                iterNum = 1

            for i in xrange(iterNum):
                self.updateParticles(tstep*0.2)

        clearGLBuffers()
        with self.viewControl.with_vp:
            self.flowVis.render()

            #with self.partVBO.array:
            #    glVertexAttribPointer(0, 4, GL_FLOAT, False, 0, 0)
            #with ctx(vattr(0), self.dustVert):
            #    glDrawArrays(GL_POINTS, self.dustOfs, self.dustN)

    def OnKeyDown(self, evt):
        key = evt.GetKeyCode()
        if key == ord(' '):
            self.paused = not self.paused
        else:
            ZglAppWX.OnKeyDown(self, evt)

if __name__ == "__main__":
    app = App()
    app.run()
