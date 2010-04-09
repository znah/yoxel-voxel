from __future__ import with_statement
import sys
sys.path.append('..')
import os
from zgl import *
from cutools import *
import pycuda.gl as cuda_gl
from voxelizer import *
from brickpool import BrickPool


def cumsum_ex(a):
    a = a.ravel()
    last = a[-1]
    b = cumsum(a)
    total = b[-1]
    b -= a
    return (b, int(total))

def profile_run(f):
    f() # warm up
    cu.Context.synchronize()
    t = clock()
    for i in xrange(100):
        f()
    cu.Context.synchronize()
    print (clock() - t) / 100.0 * 1000, 'ms'


class App(ZglAppWX):

    def __init__(self):
        ZglAppWX.__init__(self, viewControl = OrthoCamera(), zglpath='..')
        import pycuda.gl.autoinit

        brick_pool = BrickPool()

        size = 1024
        voxelizer = Voxelizer(size)
        v, f = load_obj("../t.obj")

        with voxelizer:
            drawArrays(GL_TRIANGLES, verts = v, indices = f)
        
        gl2cudaBuf = cuda_gl.BufferObject(voxelizer.dumpToPBO().handle)

        mod = SourceModule(file('density.cu').read(), include_dirs = [os.getcwd(), os.getcwd()+'/../include'], no_extern_c = True, keep = True)
        CalcDensity = mod.get_function('CalcDensity') 
        MarkBricks  = mod.get_function('MarkBricks') 
        PackBricks  = mod.get_function('PackBricks') 

        print CalcDensity.num_regs, CalcDensity.shared_size_bytes
        print MarkBricks.num_regs, MarkBricks.shared_size_bytes
        print PackBricks.num_regs, PackBricks.shared_size_bytes

        # sum 2x2x2 bits
        d_density = ga.zeros((size/2, size/2, size/2*4/32), uint32)
        gl2cudaMap = gl2cudaBuf.map()
        def f():
            CalcDensity(int32(gl2cudaMap.device_ptr()), d_density, block = (4, 8, 8), grid = (size/8/2, size/2))
        profile_run(f)
        gl2cudaMap.unmap()

        # mark and enumerate non-uniform bricks
        grid_size = size / 2 / 4
        d_bricks      = ga.zeros((grid_size, grid_size, grid_size), uint32)
        d_columnStart = ga.zeros((grid_size, grid_size), uint32)
        def f():
            MarkBricks(d_density, d_bricks, d_columnStart, block = (grid_size/2, 1, 1), grid = (grid_size, grid_size))
        profile_run(f)
        columnStart, total = cumsum_ex(d_columnStart.get())
        d_columnStart.set(columnStart)

        # pack non-uniform brick ids
        d_packedBricks = ga.zeros((total,), uint32)
        def f():
            PackBricks(d_bricks, d_columnStart, d_packedBricks, block = (grid_size, 1, 1), grid = (grid_size, grid_size))
        profile_run(f)


        brick_pool.allocMap(total)

        brick_pool.commit()

        save('mark', brick_pool.d_map_mark_enum.get())





        print total
        save('colsum', d_columnStart.get() )
        save('packed', d_packedBricks.get())

    
    #def display(self):
        #clearGLBuffers()
        #with ctx(self.viewControl.with_vp, self.fragProg(z = self.time*0.1%1.0)):  #self.z)):
        #    drawQuad()

if __name__ == "__main__":
    App().run()
    