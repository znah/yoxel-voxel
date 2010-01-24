from __future__ import with_statement
import sys
sys.path.append('..')
from zgl import *

import pycuda.driver as cu
import pycuda.gpuarray as ga
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
from cutypes import *
import os

class ReactDiff:
    def __init__(self, sz = 64):
        self.volSize = sz

        self.block = (8, 8, 4)
        self.grid = (sz / self.block[0], sz / self.block[1] * sz / self.block[2])
        print self.grid

        code = cu_header + '''
          texture<float2, 3> srcTex;

          extern "C"
          __global__ void RDKernel(float f, float k, int sz, float2 * dstBuf)
          {
            int bx = blockIdx.x;
            int volSizeY = (sz / blockDim.y);
            int by = blockIdx.y % volSizeY;
            int bz = blockIdx.y / volSizeY;

            int x = threadIdx.x + bx * blockDim.x;
            int y = threadIdx.y + by * blockDim.y;
            int z = threadIdx.z + bz * blockDim.z;

            float2 v = tex3D(srcTex, x, y, z);
            float2 l = -6.0f * v;
            l += tex3D(srcTex, x+1.0f, y, z);
            l += tex3D(srcTex, x-1.0f, y, z);
            l += tex3D(srcTex, x, y+1.0f, z);
            l += tex3D(srcTex, x, y-1.0f, z);
            l += tex3D(srcTex, x, y, z+1.0f);
            l += tex3D(srcTex, x, y, z-1.0f);

            l *= 2.0f;

            float2 diffCoef = make_float2(0.082, 0.041);
            float dt = 0.5f;

            float rate = v.x * v.y * v.y;
            float2 dv;
            dv.x = l.x * diffCoef.x - rate + f*(1.0f - v.x);
            dv.y = l.y * diffCoef.y + rate - (f + k) * v.y;
            v += dt * dv;

            int ofs = x + (y + z * sz) * sz;
            //v = make_float2(1, x + y + z);
            dstBuf[ofs] = v;
          }

        '''
        self.mod = SourceModule(code, include_dirs = [os.getcwd()], no_extern_c = True)
        

        descr = cu.ArrayDescriptor3D()
        descr.width = sz
        descr.height = sz
        descr.depth = sz
        descr.format = cu.dtype_to_array_format(float32)
        descr.num_channels = 2
        descr.flags = 0
        self.d_srcArray = cu.Array(descr)

        initArray = zeros((sz, sz, sz, 2), float32)
        initArray[...,0] = 1
        c = sz/2
        initArray[c:,c:,c:][:20,:20,:20] = (1, 1)
        self.d_dst = ga.to_gpu(initArray)
        print initArray .strides

        self.dst2src = copy = cu.Memcpy3D()
        copy.set_src_device(self.d_dst.gpudata)
        copy.set_dst_array(self.d_srcArray)
        copy.width_in_bytes = copy.src_pitch = initArray.strides[1]
        copy.src_height = copy.height = sz
        copy.depth = sz
        copy()

        self.d_srcTex = self.mod.get_texref('srcTex')
        self.d_srcTex.set_array(self.d_srcArray)

        self.RDKernel = self.mod.get_function('RDKernel')

    def iterate(self):
        self.RDKernel(float32(0.027), float32(0.062), int32(self.volSize), self.d_dst, 
          block = self.block, grid = self.grid, texrefs = [self.d_srcTex])
        self.dst2src()
       



sz = 64
rd = ReactDiff(sz = sz)
for i in xrange(3000):
    rd.iterate()
    print '.',

a = rd.d_dst.get()
b = (a[...,1]*255).astype(uint8)
b.tofile("a.dat")

import pylab
pylab.imshow(a[sz / 2 + 10, ... ,1])
pylab.colorbar()
pylab.show()



'''
    
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
    App().run()
'''