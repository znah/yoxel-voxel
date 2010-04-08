from __future__ import with_statement
import sys
sys.path.append('..')
import os
from zgl import *
from cutools import *
import pycuda.gl as cuda_gl
from voxelizer import *


def cumsum_ex(a):
    a = a.ravel()
    last = a[-1]
    b = cumsum(a)
    total = b[-1]
    b -= a
    return (b, int(total))
   


class BrickPool(HasTraits):
    brick_size = ReadOnly()
    pool_shape = ReadOnly()
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
            shape = map(int, shape)
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
        
        map_buf_shape = V(self.map_pool_size, pool_shape[1], pool_shape[2])
        d_map_idx        = ga.zeros(map(int, map_buf_shape), int32)
        d_map_bricks     = ga.zeros(map(int, map_buf_shape*brick_size), uint8)
        d_map2pool_slice = ga.zeros(self.map_pool_size, uint32)

        slice_shape =  pool_shape[1:]
        slice_capacity = prod(slice_shape)
        slice_item_count = zeros(pool_shape[0], int32)
        slice_free_mark = ones(pool_shape, int32)

        pool2map = cu.Memcpy3D()
        pool2map.set_src_array (self.d_pool_array)
        pool2map.set_dst_device(d_map_bricks.gpudata)
        pool2map.dst_pitch      = d_map_bricks.shape[2] * d_map_bricks.dtype.itemsize
        pool2map.dst_height     = d_map_bricks.shape[1]
        pool2map.width_in_bytes = pool2map.dst_pitch
        pool2map.height         = d_map_bricks.shape[1]
        pool2map.depth          = brick_size

        map2pool = cu.Memcpy3D()
        map2pool.set_dst_array (self.d_pool_array)
        map2pool.set_src_device(d_map_bricks.gpudata)
        map2pool.src_pitch      = d_map_bricks.shape[2] * d_map_bricks.dtype.itemsize
        map2pool.src_height     = d_map_bricks.shape[1]
        map2pool.width_in_bytes = map2pool.src_pitch
        map2pool.height         = d_map_bricks.shape[1]
        map2pool.depth          = brick_size

        class PoolMapping:
            pass

        def alloc_map(n):
            order = argsort(slice_item_count)
            allocated = 0
            i = 0
            map2pool_slice = zeros(self.map_pool_size, uint32)
            map_idx        = zeros(map_buf_shape, int32)

            while allocated < n and i < self.map_pool_size:
               sidx = order[i]
               free_count = slice_capacity - slice_item_count[sidx]
               to_alloc = min(free_count, n - allocated)
               
               midx = map_idx[i]
               midx = cumsum_ex(slice_free_mark[sidx])[0].reshape(slice_shape) + allocated
               midx[slice_free_mark[sidx] == 0] = -1
               midx[midx >= n] = -1
               slice_free_mark[sidx][midx >= 0] = 0
               slice_item_count[sidx] += to_alloc

               pool2map.srcZ = sidx * brick_size
               pool2map.dstZ = i * brick_size
               pool2map()
               map2pool_slice[i] = sidx

               allicated += to_alloc
               i += 1

            d_map_idx.set(map_idx)
            d_map2pool_slice.set(map2pool_slice)
            self.item_count += allicated

            mapping = PoolMapping()
            mapping.d_map_idx        = d_map_idx
            mapping.map2pool_slice   = map2pool_slice
            mapping.d_map2pool_slice = d_map2pool_slice
            mapping.d_map_bricks     = d_map_bricks
            mapping.mapped_count = allocated
            mapping.slice_count  = i

            def commit():
                for i in xrange(slice_count):
                   map2pool.srcZ = i * brick_size
                   map2pool.dstZ = map2pool_slice[i] * brick_size
                   map2pool()
            mapping.commit = commit
        
        self.alloc_map = alloc_map


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


        mapping = brick_pool.alloc_map(total)

        brick_pool.unmap()





        print total
        save('colsum', d_columnStart.get() )
        save('packed', d_packedBricks.get())

    
    #def display(self):
        #clearGLBuffers()
        #with ctx(self.viewControl.with_vp, self.fragProg(z = self.time*0.1%1.0)):  #self.z)):
        #    drawQuad()

if __name__ == "__main__":
    App().run()
    