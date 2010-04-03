from __future__ import with_statement
from zgl import *
from voxelizer import *


code = '''


__global__ 


'''
    
class App(ZglAppWX):
    z = Range(0.0, 1.0, 0.5)

    def __init__(self):
        ZglAppWX.__init__(self, viewControl = OrthoCamera())
        import pycuda.gl.autoinit

        size = 1024
        voxelizer = Voxelizer(size)
        v, f = load_obj("t.obj")
        with voxelizer:
            drawArrays(GL_TRIANGLES, verts = v, indices = f)
        
        gl2cudaBuf = cuda_gl.BufferObject(self.voxelizer.dumpToPBO().handle)

        



        

    
    def display(self):
        clearGLBuffers()
        with ctx(self.viewControl.with_vp, self.fragProg(z = self.time*0.1%1.0)):  #self.z)):
            drawQuad()

if __name__ == "__main__":
    App().run()
