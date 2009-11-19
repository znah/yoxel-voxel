from __future__ import with_statement
from zgl import *


class App(ZglApp):
    def __init__(self):
        ZglApp.__init__(self, OrthoCamera())

        vortexRowN = 4
        dustRowN = 256
        w = 256
        self.psize = (w, vortexRowN + dustRowN)
        self.vortexOfs = 0
        self.vortexN = vortexRowN * w
        self.dustOfs = self.vortexOfs + self.vortexN
        self.dustN = dustRowN * w

        pstate = random.rand(self.psize[1], self.psize[0], 4).astype(float32)
        pstate[..., 2] = pstate[..., 2] ** 5
        pstate[..., 2] = 0.02 + (pstate[..., 2])*1  # radius
        pstate[..., 3] = (2 * pstate[..., 3] - 1)*3 # curl
        self.particles = PingPong(img=pstate, format = GL_RGBA_FLOAT32_ATI)
        self.partVBO = BufferObject(pstate, GL_STREAM_COPY)

        flowBufSize = 256
        self.flowTex = RenderTexture(size = (flowBufSize, flowBufSize), format = GL_RGBA_FLOAT16_ATI)
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

        self.visFrag = CGShader( 'fp40', '''
          uniform sampler2D flowTex;
          uniform sampler2D sandTex;
          float4 main( float2 tc: TEXCOORD0 ) : COLOR
          {
            return tex2D(sandTex, tc);
          }
        ''')
        
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

        a = zeros((512, 512, 4), float32)
        a[100:200,100:200,0] = 1
        self.sandTex = PingPong(img=a, format = GL_RGBA_FLOAT16_ATI)
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
        self.sandUpdate.flowTex = self.flowTex.tex

        glTexEnvf(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)

        self.time = clock()

    def updateParticles(self, dt):
        glEnable(GL_POINT_SPRITE)
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB)

        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE);
        glBlendEquation(GL_FUNC_ADD);

        with ctx(self.flowTex, ortho, self.vortexVert, self.vortexFrag):
            glClearColor(0, 0, 0, 0)
            glClear(GL_COLOR_BUFFER_BIT)
            with self.partVBO.array:
                glVertexAttribPointer(0, 4, GL_FLOAT, False, 0, 0)
            with vattr(0):
                glDrawArrays(GL_POINTS, self.vortexOfs, self.vortexN)
        
        glDisable(GL_BLEND)
        glDisable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB)
        glDisable(GL_POINT_SPRITE)
            
        self.partUpdateProg.dt = dt
        self.partUpdateProg.partTex = self.particles.src.tex
        self.partUpdateProg.flowTex = self.flowTex.tex
        with ctx(self.particles.dst, ortho, self.partUpdateProg):
            drawQuad()
            with self.partVBO.pixelPack:
                OpenGL.raw.GL.glReadPixels(0, 0, self.psize[0], self.psize[1], GL_RGBA, GL_FLOAT, None)
        self.particles.flip()

        self.sandUpdate.sandTex = self.sandTex.src.tex
        self.sandUpdate.dt = dt
        with ctx(self.sandTex.dst, self.sandUpdate, ortho):
            drawQuad()
        self.sandTex.flip()

    
    def display(self):
        t = clock()
        dt = t - self.time
        self.time = t

        self.updateParticles(dt)

        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self.visFrag.flowTex = self.flowTex.tex
        self.visFrag.sandTex = self.sandTex.src.tex
        with self.viewControl.with_vp:
            with self.visFrag:
                drawQuad()

            #with self.partVBO.array:
            #    glVertexAttribPointer(0, 4, GL_FLOAT, False, 0, 0)
            #with ctx(vattr(0), self.dustVert):
            #    glDrawArrays(GL_POINTS, self.dustOfs, self.dustN)

        glutSwapBuffers()

if __name__ == "__main__":
  viewSize = (800, 600)
  zglInit(viewSize, "hello")

  glutSetCallbacks(App())

  #wglSwapIntervalEXT(0)
  glutMainLoop()
