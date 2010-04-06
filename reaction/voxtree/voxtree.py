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

        mod = SourceModule(file('density.cu').read(), include_dirs = [os.getcwd(), os.getcwd()+'/../include'], no_extern_c = True, keep = True)
        CalcDensity = mod.get_function('CalcDensity') 
        print CalcDensity.num_regs, CalcDensity.shared_size_bytes

        d_density = ga.zeros((size/2, size/2, size/2*4/32), uint32)

        gl2cudaMap = gl2cudaBuf.map()
        CalcDensity(int32(gl2cudaMap.device_ptr()), d_density, block = (4, 8, 8), grid = (size/8/2, size/2))
        gl2cudaMap.unmap()

        grid_size = size / 2 / 4
        MarkBricks = mod.get_function('MarkBricks') 
        print MarkBricks.num_regs, MarkBricks.shared_size_bytes

        d_bricks = ga.zeros((grid_size, grid_size, grid_size), uint32)
        d_colsum  = ga.zeros((grid_size, grid_size), uint32)
        def f():
          MarkBricks(d_density, d_bricks, d_colsum, block = (grid_size/2, 1, 1), grid = (grid_size, grid_size))
          colsum = d_colsum.get().ravel()
          cumsum(colsum[1:], out = colsum[1:])
          total = colsum[0] + colsum[-1]
          colsum[0] = 0
          d_colsum.set(colsum)
          return total
        print f()

        cu.Context.synchronize()
        t = clock()
        for i in xrange(100):
            f()
        cu.Context.synchronize()
        print (clock() - t) / 100.0 * 1000

        save('colsum',  d_colsum.get() )
    
    #def display(self):
        #clearGLBuffers()
        #with ctx(self.viewControl.with_vp, self.fragProg(z = self.time*0.1%1.0)):  #self.z)):
        #    drawQuad()

if __name__ == "__main__":
    App().run()
