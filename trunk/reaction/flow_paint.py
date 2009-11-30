from __future__ import with_statement
from zgl import *


class App(ZglApp):
    def __init__(self):
        ZglApp.__init__(self, OrthoCamera())

        self.pstateUpdateFP = CGShader("fp40", fileName = "flow_paint.cg", entry = "pstateUpdateFP")
        self.vortexVP = CGShader("vp40", fileName = "flow_paint.cg", entry = "vortexVP")
        self.vortexFP = CGShader("fp40", fileName = "flow_paint.cg", entry = "vortexFP")
        self.shaders = [self.pstateUpdateFP, self.vortexVP, self.vortexFP]

        pstate = random.rand(256, 256, 4).astype(float32)
        self.pstate = PingPong(img=pstate, format = GL_RGBA_FLOAT16_ATI)
        self.flow = RenderTexture(size = (512, 384),format = GL_RGBA_FLOAT16_ATI)

        self.setCommonParam("flowTex", self.flow.tex)
        self.setCommonParam("vortexScale", self.flow.size()[1]) # y size
        self.setCommonParam("dt", 0.01)

        glTexEnvf(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB)
        glClearColor(0, 0, 0, 0)

        with ctx(self.flow, ortho, self.vortexVP, self.vortexFP, glstate(GL_POINT_SPRITE)):
            glClear(GL_COLOR_BUFFER_BIT)
            glBegin(GL_POINTS)
            glVertex(0.5, 0.5, 0.5, 1)
            glEnd()
            a = glReadPixels(0, 0, 512, 384, GL_RGBA, GL_FLOAT)
            a.shape = (384, 512, 4)
            save("a", a)

    def setCommonParam(self, name, val):
        for prog in self.shaders:
            setattr(prog, name, val)

    def display(self):

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        

        glutSwapBuffers()

if __name__ == "__main__":
  viewSize = (800, 600)
  zglInit(viewSize, "hello")

  glutSetCallbacks(App())

  #wglSwapIntervalEXT(0)
  glutMainLoop()
