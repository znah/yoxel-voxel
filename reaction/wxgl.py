from __future__ import with_statement
from zgl import *
    
class App(ZglAppWX):
    def __init__(self):
        ZglAppWX.__init__(self, viewControl = OrthoCamera())
        self.fragProg = CGShader('fp40', TestShaders, entry = 'TexCoordFP')
    
    def display(self):
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        with ctx(self.viewControl.with_vp, self.fragProg):
            drawQuad()

if __name__ == "__main__":
    app = App()
    app.run()