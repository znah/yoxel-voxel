from __future__ import with_statement
from zgl import *

class App(ZglApp):
    def __init__(self):
        ZglApp.__init__(self, OrthoCamera())

        self.psize = (64, 64)
        pstate = random.rand(self.psize[0], self.psize[1], 2).astype(float32)
        pstate = tile(pstate, (1, 1, 2))
        self.particles = PingPong(img=pstate, format = GL_RGBA_FLOAT32_ATI)
        self.partVBO = BufferObject(pstate, GL_STREAM_COPY)

        self.flowTex = RenderTexture(size = (512, 512), format = GL_RGBA_FLOAT16_ATI)

        self.partUpdateProg = CGShader( 'fp40', '''
          uniform sampler2D partTex;
          uniform float t;
          uniform float dt;

          float4 main( float2 tc: TEXCOORD0 ) : COLOR
          {
            float4 data = tex2D(partTex, tc);
            float2 p = data.xy;
            p += float2(0.1, 0.03) * dt;
            p = frac(p + float2(1, 1));
            return float4(p, 0, 0);
          }
        ''')

        self.fragProg = CGShader('fp40', '''
          float4 main( float2 tc: TEXCOORD0 ) : COLOR 
          { 
            return float4(1, 1, 1, 1); 
          }
        ''')
        
        self.vertProg = CGShader('vp40', '''
          float4 main( float4 pos: ATTR0) : POSITION
          { 
            float4 p = float4(pos.xy, 0, 1);
            return mul(glstate.matrix.mvp, p); 
          }
        ''')

    def updateParticles(self):
        self.partUpdateProg.dt = 0.01
        self.partUpdateProg.partTex = self.particles.src.tex
        with ctx(self.particles.dst, ortho, self.partUpdateProg):
            drawQuad()
            with self.partVBO.pixelPack:
                OpenGL.raw.GL.glReadPixels(0, 0, self.psize[0], self.psize[1], GL_RGBA, GL_FLOAT, None)
        self.particles.flip()

    
    def display(self):
        self.updateParticles()

        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        #self.fragProg.tex = self.particles.src.tex
        with ctx(self.viewControl.with_vp, self.vertProg, self.fragProg):
            with self.partVBO.array:
                glVertexAttribPointer(0, 4, GL_FLOAT, False, 0, 0)
            with vattr(0):
                glDrawArrays(GL_POINTS, 0, prod(self.psize))


        glutSwapBuffers()

if __name__ == "__main__":
  viewSize = (800, 600)
  zglInit(viewSize, "hello")

  glutSetCallbacks(App())

  #wglSwapIntervalEXT(0)
  glutMainLoop()
