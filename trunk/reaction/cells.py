from __future__ import with_statement
from zgl import *

class App(ZglAppWX):
    def __init__(self):
        ZglAppWX.__init__(self, viewControl = OrthoCamera())

        self.cellFrag = CGShader("fp40", fileName = "cells.cg")

        self.noiseTex = Texture2D(random.rand(512, 512, 4).astype(float32), format=GL_RGBA_FLOAT32_ATI)
        self.cellFrag.noiseTex = self.noiseTex
        self.cellFrag.noiseSize = self.noiseTex.size
    
    def display(self):
        
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self.cellFrag.time = self.time
        with ctx(self.viewControl.with_vp, self.cellFrag):
            drawQuad()

if __name__ == "__main__":
    app = App()
    app.run()
