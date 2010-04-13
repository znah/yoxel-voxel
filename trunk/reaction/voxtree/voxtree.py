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
    z = Range(0, 512, 100)

    @on_trait_change('z')
    def updateSlice(self):
        pass
        


    def __init__(self):
        ZglAppWX.__init__(self, viewControl = OrthoCamera(), zglpath='..')
        import pycuda.gl.autoinit

        brick_pool = BrickPool()

        size = 1024
        voxelizer = Voxelizer(size)
        v, f = load_obj("../t.obj")

        with voxelizer:
            clearGLBuffers()
            drawArrays(GL_TRIANGLES, verts = v, indices = f)
        
        gl2cudaBuf = cuda_gl.BufferObject(voxelizer.dumpToPBO().handle)

        mod = SourceModule(file('density.cu').read(), include_dirs = [os.getcwd(), os.getcwd()+'/../include'], no_extern_c = True, keep = True)
        CalcDensity = mod.get_function('CalcDensity') 
        MarkBricks  = mod.get_function('MarkBricks') 
        PackBricks  = mod.get_function('PackBricks') 
        UpdateBrickPool  = mod.get_function('UpdateBrickPool') 
        FetchTest   = mod.get_function('FetchTest') 
        d_slot2slice = mod.get_global('slot2slice')[0]
        brick_pool_tex = brick_pool_tex = mod.get_texref('brick_pool_tex')
        brick_grid_tex = brick_grid_tex = mod.get_texref('brick_grid_tex')

        print CalcDensity.num_regs, CalcDensity.shared_size_bytes
        print MarkBricks.num_regs, MarkBricks.shared_size_bytes
        print PackBricks.num_regs, PackBricks.shared_size_bytes
        print UpdateBrickPool.num_regs, UpdateBrickPool.shared_size_bytes
        print FetchTest.num_regs, FetchTest.shared_size_bytes

        # sum 2x2x2 bits
        d_density = ga.zeros((size/2, size/2, size/2*4/32), uint32)
        gl2cudaMap = gl2cudaBuf.map()
        def f():
            CalcDensity(int32(gl2cudaMap.device_ptr()), d_density, block = (4, 8, 8), grid = (size/8/2, size/2))
        profile_run(f)
        gl2cudaMap.unmap()

        save("coral", d_density.get())
        return

        # mark and enumerate non-uniform bricks
        grid_size = size / 2 / 4
        d_bricks = d_bricks = ga.zeros((grid_size, grid_size, grid_size), uint32)
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

        # update brick pool
        brick_pool.allocMap(total)
        cu.memcpy_htod(d_slot2slice, brick_pool.slot2slice)
        
        def f():
            UpdateBrickPool(
              int32(brick_pool.pool_shape[2]), 
              brick_pool.d_map_mark_enum,
              d_packedBricks,
              d_density,
              brick_pool.d_map_slots,
              d_bricks,
              block = (5, 5, 5),
              grid = (brick_pool.pool_shape[2] * len(brick_pool.slot2slice), brick_pool.pool_shape[1]))
        profile_run(f)

        #save("slots", brick_pool.d_map_slots.get())
        brick_pool.commit()

        # test fetch
        cu.bind_array_to_texref( brick_pool.d_pool_array, brick_pool_tex )
        d_bricks.bind_to_texref( brick_grid_tex )
        self.brick_pool = brick_pool # PROTECT FROM GARBAGE COLLECTION
        self.d_bricks = d_bricks     # PROTECT FROM GARBAGE COLLECTION

        brick_pool_tex.set_filter_mode( cu.filter_mode.LINEAR )

        d_res = ga.zeros((1024, 1024, 4), uint8)
        tex = Texture2D(size = (1024, 1024))
        texFrag = genericFP('tex2D(texture, tc0.xy)')
        texFrag.texture = tex

        def updateSlice():
            with cuprofile('updateSlice'):
                FetchTest(float32(0), float32(0), float32(self.z), d_res, block=(16, 16, 1), grid = (64, 64), 
                    texrefs = [brick_pool_tex, brick_grid_tex])
            with tex:
                glTexSubImage2D(tex.Target, 0, 0, 0, 1024, 1024, GL_RGBA, GL_UNSIGNED_BYTE, d_res.get())
            self.fp = texFrag
        self.updateSlice = updateSlice
        updateSlice()

        print total
    
    def display(self):
        clearGLBuffers()

        with ctx(self.viewControl.with_vp, self.fp):
            drawQuad()

if __name__ == "__main__":
    App().run()
    