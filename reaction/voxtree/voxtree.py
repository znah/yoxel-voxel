from __future__ import with_statement
import sys
sys.path.append('..')
import os
from zgl import *
from cutools import *
import pycuda.gl as cuda_gl
from voxelizer import *


def cumsum_ex(a):
    last = a[-1]
    b = cumsum(a)
    total = b[-1]
    b -= a
    return (b, int(total))
   


'''
class BrickPool(HasTraits):
    brick_size = ReadOnly()
    pool_shape  = ReadOnly()
    capacity   = ReadOnly

    map_pool_size = ReadOnly(16)

    d_pool_array = ReadOnly()

    item_count = Int(0)

    _ = Python(editable = False)
    
    def __init__(self, brick_size = 5, pool_shape = (96, 96, 96)):
        self.brick_size = brick_size
        self.pool_shape = pool_shape
        self.capacity   = prod(pool_shape)

        def createArray(shape):
            descr = cu.ArrayDescriptor3D()
            descr.width  = shape[2]
            descr.height = shape[1]
            descr.depth  = shape[0]
            descr.format = cu.dtype_to_array_format(uint8)
            descr.num_channels = 1
            descr.flags = 0
            return cu.Array(descr)

        pool_array_shape = V(pool_shape) * brick_size
        self.d_pool_array = createArray( pool_array_shape )
        
        map_buf_shape = V(self.map_pool_size, pool_shape[1], pool_shape[2]) * brick_size
        d_map_buf = ga.zeros(map_buf_shape, uint8)

        empty_map = ones(pool_shape, int32)

        level_capasity = prod(pool_shape[1:])
        level_item_count = zeros(pool_shape[0], int32)


        d_mapped
        d_src_index



    def alloc_map(self, n):
        
        
'''


class App(ZglAppWX):

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
        MarkBricks  = mod.get_function('MarkBricks') 
        PackBricks  = mod.get_function('PackBricks') 

        print CalcDensity.num_regs, CalcDensity.shared_size_bytes
        print MarkBricks.num_regs, MarkBricks.shared_size_bytes
        print PackBricks.num_regs, PackBricks.shared_size_bytes

        # sum 2x2x2 bits
        d_density = ga.zeros((size/2, size/2, size/2*4/32), uint32)
        gl2cudaMap = gl2cudaBuf.map()
        CalcDensity(int32(gl2cudaMap.device_ptr()), d_density, block = (4, 8, 8), grid = (size/8/2, size/2))
        gl2cudaMap.unmap()

        # mark and enumerate non-uniform bricks
        grid_size = size / 2 / 4
        d_bricks      = ga.zeros((grid_size, grid_size, grid_size), uint32)
        d_columnStart = ga.zeros((grid_size, grid_size), uint32)
        MarkBricks(d_density, d_bricks, d_columnStart, block = (grid_size/2, 1, 1), grid = (grid_size, grid_size))
        columnStart = d_columnStart.get().ravel()
        columnStart, total = cumsum_ex(columnStart)
        d_columnStart.set(columnStart)
        print total

        # pack non-uniform brick ids
        d_packedBricks = ga.zeros((total,), uint32)
        PackBricks(d_bricks, d_columnStart, d_packedBricks, block = (grid_size, 1, 1), grid = (grid_size, grid_size))



        '''
        cu.Context.synchronize()
        t = clock()
        for i in xrange(100):
            f()
        cu.Context.synchronize()
        print (clock() - t) / 100.0 * 1000
        '''
        
        save('colsum', d_columnStart.get() )
        save('packed', d_packedBricks.get())

    
    #def display(self):
        #clearGLBuffers()
        #with ctx(self.viewControl.with_vp, self.fragProg(z = self.time*0.1%1.0)):  #self.z)):
        #    drawQuad()

if __name__ == "__main__":
    App().run()
    