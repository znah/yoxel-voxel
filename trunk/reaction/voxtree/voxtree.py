from __future__ import with_statement
import sys
sys.path.append('..')
import os
from zgl import *
from cutools import *
import pycuda.gl as cuda_gl
from voxelizer import *


class App(ZglAppWX):
    z = Range(0.0, 1.0, 0.5)

    def __init__(self):
        ZglAppWX.__init__(self, viewControl = OrthoCamera(), zglpath='..')
        import pycuda.gl.autoinit

        size = 1024
        voxelizer = Voxelizer(size)
        v, f = load_obj("../t.obj")

        with voxelizer:
            drawArrays(GL_TRIANGLES, verts = v, indices = f)
        
        gl2cudaBuf = cuda_gl.BufferObject(voxelizer.dumpToPBO().handle)

        mod = SourceModule(file('density.cu').read(), include_dirs = [os.getcwd(), os.getcwd()+'/../include'], no_extern_c = True)
        CalcDensity = mod.get_function('CalcDensity') 
        print CalcDensity.num_regs, CalcDensity.shared_size_bytes

        d_density = ga.zeros((size/2, size/2, size/2*4/32), uint32)

        gl2cudaMap = gl2cudaBuf.map()
        CalcDensity(int32(gl2cudaMap.device_ptr()), d_density, block = (4, 8, 8), grid = (size/8/2, size/2))
        gl2cudaMap.unmap()

        grid_size = size / 2 / 4
        MarkBricks = mod.get_function('MarkBricks') 
        print MarkBricks.num_regs, MarkBricks.shared_size_bytes

        mixedBits = ga.zeros((grid_size, grid_size, grid_size/32), uint32)
        solidBits = ga.zeros((grid_size, grid_size, grid_size/32), uint32)

        MarkBricks(d_density, mixedBits, solidBits, block = (64, 1, 1), grid = (grid_size, grid_size))

        save('mixed', mixedBits.get() )
        save('solid', solidBits.get() )
        save("density", d_density.get())
    
    #def display(self):
        #clearGLBuffers()
        #with ctx(self.viewControl.with_vp, self.fragProg(z = self.time*0.1%1.0)):  #self.z)):
        #    drawQuad()

if __name__ == "__main__":
    App().run()
