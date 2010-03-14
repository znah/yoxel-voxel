from __future__ import with_statement
import sys
sys.path.append('..')

from zgl import *
from volvis import VolumeRenderer
from cutools import *
import os

from enthought.traits.ui.api import *
class ReactDiff(HasTraits):
    f = Float(0.025) #0.27
    k = Float(0.062)
    scale = Range(0.5, 2.0, 1.0)
    reset = Button(label = "Reset")

    view = View(
        Item( 'f' ),
        Item( 'k' ),
        Item( 'scale' ),
        Item( 'reset' ))


    _ = Python(editable = False)

    def _reset_fired(self):
        self.d_dst.set(self.initArray)
        self.dst2src()

    def __init__(self, sz = 64):
        self.volSize = sz

        self.block = (8, 8, 4)
        self.grid = (sz / self.block[0], sz / self.block[1] * sz / self.block[2])
        print self.grid

        code = cu_header + '''
          texture<float2, 3> srcTex;

          extern "C"
          __global__ void RDKernel(float f, float k, float scale, int sz, float2 * dstBuf)
          {
            int bx = blockIdx.x;
            int volSizeY = (sz / blockDim.y);
            int by = blockIdx.y % volSizeY;
            int bz = blockIdx.y / volSizeY;

            int x = threadIdx.x + bx * blockDim.x;
            int y = threadIdx.y + by * blockDim.y;
            int z = threadIdx.z + bz * blockDim.z;

            float dp = 1.0f / sz;
            float3 p = make_float3(x, y, z) * dp;

            float2 v = tex3D(srcTex, p.x, p.y, p.z);
            float2 l = -6.0f * v;
            l += tex3D(srcTex, p.x+dp, p.y,    p.z);
            l += tex3D(srcTex, p.x-dp, p.y,    p.z);
            l += tex3D(srcTex, p.x,    p.y+dp, p.z);
            l += tex3D(srcTex, p.x,    p.y-dp, p.z);
            l += tex3D(srcTex, p.x,    p.y,    p.z+dp);
            l += tex3D(srcTex, p.x,    p.y,    p.z-dp);

            l *= scale;

            float2 diffCoef = make_float2(0.082, 0.041);
            float dt = 0.5f;

            float rate = v.x * v.y * v.y;
            float2 dv;
            dv.x = l.x * diffCoef.x - rate + f*(1.0f - v.x);
            dv.y = l.y * diffCoef.y + rate - (f + k) * v.y;
            v += dt * dv;

            int ofs = x + (y + z * sz) * sz;
            dstBuf[ofs] = v;
          }

        '''
        self.mod = SourceModule(code, include_dirs = [os.getcwd(), os.getcwd()+'/../include'], no_extern_c = True)
        

        descr = cu.ArrayDescriptor3D()
        descr.width = sz
        descr.height = sz
        descr.depth = sz
        descr.format = cu.dtype_to_array_format(float32)
        descr.num_channels = 2
        descr.flags = 0
        self.d_srcArray = cu.Array(descr)

        self.initArray = initArray = zeros((sz, sz, sz, 2), float32)
        initArray[...,0] = 1
        
        for i in xrange(2):
            p = random.rand(3) * sz - 2
            initArray[p[0]:, p[1]:, p[2]:,1][:2, :2, :2] = 1
        self.d_dst = ga.to_gpu(initArray)

        self.dst2src = copy = cu.Memcpy3D()
        copy.set_src_device(self.d_dst.gpudata)
        copy.set_dst_array(self.d_srcArray)
        copy.width_in_bytes = copy.src_pitch = initArray.strides[1]
        copy.src_height = copy.height = sz
        copy.depth = sz
        copy()

        self.d_srcTex = self.mod.get_texref('srcTex')
        self.d_srcTex.set_array(self.d_srcArray)
        self.d_srcTex.set_flags(cu.TRSF_NORMALIZED_COORDINATES)
        self.d_srcTex.set_address_mode(0, cu.address_mode.WRAP)
        self.d_srcTex.set_address_mode(1, cu.address_mode.WRAP)
        self.d_srcTex.set_address_mode(2, cu.address_mode.WRAP)

        self.RDKernel = self.mod.get_function('RDKernel')

    @with_(cuprofile("RDStep"))
    def iterate(self):
        self.RDKernel(float32(self.f), float32(self.k), float32(self.scale), int32(self.volSize), self.d_dst, 
          block = self.block, grid = self.grid, texrefs = [self.d_srcTex])
        self.dst2src()
       


def sync_clock():
    cu.Context.synchronize()
    return clock()


class App(ZglAppWX):
    volumeRender = Instance(VolumeRenderer)    
    rd = Instance(ReactDiff)    
    iterPerFrame = Int(20)
    
    def __init__(self):
        ZglAppWX.__init__(self, viewControl = FlyCamera(), zglpath='..')
        
        self.rd = ReactDiff(64)
        a = self.rd.d_dst.get()
        self.volumeRender = VolumeRenderer(Texture3D(img = a[...,1]))

    def display(self):
        clearGLBuffers()
        
        for i in xrange(self.iterPerFrame):
            self.rd.iterate()
        a = self.rd.d_dst.get()
        a = ascontiguousarray(a[...,1])
        with self.volumeRender.volumeTex:
            glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, a.shape[0], a.shape[1], a.shape[2], GL_LUMINANCE, GL_FLOAT, a)
            
        with ctx(self.viewControl.with_vp):
            self.volumeRender.render()


if __name__ == "__main__":
    import pycuda.autoinit
    App().run()


'''
if __name__ == '__main__':

    sz = 64
    iterNum = 3000

    rd = ReactDiff(sz = sz)
    times = zeros((iterNum,), float64)
    for i in xrange(iterNum):
        t = sync_clock()
        rd.iterate()
        times[i] = sync_clock() - t
        if i % 100 == 0:
            print '.',
    print

    a = rd.d_dst.get()
    b = (a[...,1]*255).astype(uint8)
    b.tofile("a.dat")

    print "avg time: %2f ms" % (times.mean() * 1000 ,)
'''
