from __future__ import with_statement
from zgl import *


class App(ZglApp):
    def __init__(self):
        ZglApp.__init__(self, OrthoCamera())

        self.psize = (64, 32)
        self.vortexNum = 2
        
        pstate = random.rand(self.psize[1], self.psize[0], 4).astype(float32)
        pstate[..., 3] = (2 * pstate[..., 3] - 1)*1
        self.particles = PingPong(img=pstate, format = GL_RGBA_FLOAT32_ATI)
        self.partVBO = BufferObject(pstate, GL_STREAM_COPY)

        self.flowTex = RenderTexture(size = (512, 512), format = GL_RGBA_FLOAT16_ATI)
        self.flowTex.tex.filterLinear()

        self.partUpdateProg = CGShader( 'fp40', '''
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
        ''')

        self.vortexFrag = CGShader('fp40', '''
          float4 main( float2 tc: TEXCOORD0, float curl : TEXCOORD1 ) : COLOR 
          { 
            float2 p = tc * 2.0 - 1.0;
            float2 t = float2(-p.y, p.x);
            
            float r = 3*length(p);
            t *= exp(-r*r) * curl;

            return float4(t, 0, 1); 
          }
        ''')
        
        self.vortexVert = CGShader('vp40', '''
          void main( 
            float4 data  : ATTR0,
            out float4 oPos  : POSITION,
            out float oPSize : PSIZE,
            out float oCurl  : TEXCOORD1 ) 
          { 
            float4 p = float4(data.xy, 0, 1);
            oPos = mul(glstate.matrix.mvp, p); ;
            oPSize = 10 + data.z*100;
            oCurl = data.w;
          }
        ''')

        self.visFrag = CGShader( 'fp40', '''
          uniform sampler2D flowTex;
          float4 main( float2 tc: TEXCOORD0 ) : COLOR
          {
            float2 v = tex2D(flowTex, tc).xy;
            float vel = length(v)*3;
            return float4(float3(vel), 1);
          }
        ''')

        glTexEnvf(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)

        self.time = clock()

    def updateParticles(self, dt):
        glEnable(GL_POINT_SPRITE)
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB)
        glPointSize(16)

        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE);
        glBlendEquation(GL_FUNC_ADD);

        with ctx(self.flowTex, ortho, self.vortexVert, self.vortexFrag):
            glClearColor(0, 0, 0, 0)
            glClear(GL_COLOR_BUFFER_BIT)
            with self.partVBO.array:
                glVertexAttribPointer(0, 4, GL_FLOAT, False, 0, 0)
            with vattr(0):
                glDrawArrays(GL_POINTS, 0, prod(self.psize))
        
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

    
    def display(self):
        t = clock()
        dt =  t - self.time
        self.time = t

        self.updateParticles(dt)

        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self.visFrag.flowTex = self.flowTex.tex
        with ctx(self.viewControl.with_vp, self.visFrag):
            drawQuad()

        glutSwapBuffers()

if __name__ == "__main__":
  viewSize = (800, 600)
  zglInit(viewSize, "hello")

  glutSetCallbacks(App())

  #wglSwapIntervalEXT(0)
  glutMainLoop()
