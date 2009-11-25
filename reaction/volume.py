from __future__ import with_statement
from zgl import *

class App(ZglApp):
    def __init__(self):
        ZglApp.__init__(self, OrthoCamera())

        data = fromfile("img/bonsai.raw", uint8)
        data.shape = (256, 256, 256)
        slice_ = data[128]
        self.tex =  Texture2D(img=slice_)
        self.fragProg = CGShader('fp40', TestShaders, entry = 'TexLookupFP')
        self.fragProg.tex = self.tex
    
    def display(self):
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        with ctx(self.viewControl.with_vp, self.fragProg):
            drawQuad()

        glutSwapBuffers()

if __name__ == "__main__":
    viewSize = (800, 600)
    zglInit(viewSize, "hello")

    glutSetCallbacks(App())

    #wglSwapIntervalEXT(0)
    glutMainLoop()
