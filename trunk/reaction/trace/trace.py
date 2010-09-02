import sys
sys.path.append('..')

from zgl import *
from cutools import *
import pycuda.gl as cuda_gl
import os

from numpy.linalg import inv


class CuGLBuf:
    def __init__(self, data):
        self.gl = BufferObject(data = data)
        self.cu = cuda_gl.BufferObject(self.gl.handle)
        
def cu_compile(code):
    return SourceModule(code, include_dirs = [os.getcwd(), os.getcwd() + '/../include'], no_extern_c = True, keep=True)

def to_cuda_array3d(a):
    chnum = 1
    if a.ndim == 4:
        chnum = a.shape[3]
    descr = setattrs(cu.ArrayDescriptor3D(),
        width  = a.shape[2],
        height = a.shape[1],
        depth  = a.shape[0],
        format = cu.dtype_to_array_format(a.dtype),
        num_channels = chnum )
    d_array = cu.Array(descr)

    copy = cu.Memcpy3D()
    copy.set_src_host(a)
    copy.set_dst_array(d_array)
    copy.width_in_bytes = copy.src_pitch = a.strides[1]
    copy.src_height = copy.height = a.shape[1]
    copy.depth = a.shape[0]
    copy()

    return d_array





class CuTracer(HasTraits):
    def __init__(self):
        mod = cu_compile( file('trace.cu').read() )
        Trace        = mod.get_function('Trace')
        print "reg: %d,  lmem: %d " % (Trace.num_regs, Trace.local_size_bytes)


        c_viewSize    = mod.get_global('c_viewSize')
        c_proj2wldMtx = mod.get_global('c_proj2wldMtx')
        c_eyePos      = mod.get_global('c_eyePos')

        w, h = 1024, 768
        self.a = a = zeros((h, w, 4), uint8)
        viewTex = Texture2D(img = a)
        viewBuf  = CuGLBuf(a)

        vol = load('bonsai.npy')
        mark = load('mark.npy') 
        self.d_vol = d_vol = to_cuda_array3d(vol)  # protect from gc
        self.d_mark = d_mark = to_cuda_array3d(mark)  # protect from gc


        vol_tex = mod.get_texref('volumeTex')
        vol_tex.set_array( d_vol )
        vol_tex.set_flags( cu.TRSF_NORMALIZED_COORDINATES )
        vol_tex.set_filter_mode( cu.filter_mode.LINEAR )
        
        mark_tex = mod.get_texref('markTex')
        mark_tex.set_array( d_mark )
        mark_tex.set_flags( cu.TRSF_NORMALIZED_COORDINATES )

        #vol_tex.set_address_mode(0, cu.address_mode.WRAP)
        #vol_tex.set_address_mode(1, cu.address_mode.WRAP)
        #vol_tex.set_address_mode(2, cu.address_mode.WRAP)

        def render():
            modelview = mat( glGetFloatv(GL_MODELVIEW_MATRIX).T  )
            proj      = mat( glGetFloatv(GL_PROJECTION_MATRIX).T )

            mvp = proj * modelview
            proj2wld = inv(mvp)
            proj2wld = ascontiguousarray(proj2wld, float32)
            eyePos = array(inv(modelview)[:3, 3]).ravel()

            cu.memcpy_htod(c_viewSize[0],    int2(w, h))
            cu.memcpy_htod(c_proj2wldMtx[0], proj2wld)
            cu.memcpy_htod(c_eyePos[0],      eyePos)

            mapping = viewBuf.cu.map()
            with cuprofile('Trace'):
                Trace(int32(mapping.device_ptr()), block = (32, 8, 1), grid = (w/32, h/8),
                  texrefs = [vol_tex, mark_tex])  # mapping.size()
            mapping.unmap()
            with ctx(viewTex, viewBuf.gl.pixelUnpack):
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, None)
            return viewTex
        self.render = render


class App(ZglAppWX):
    renderer = Instance(CuTracer)    
    
    def __init__(self):
        ZglAppWX.__init__(self, viewControl = FlyCamera(), size = (1024, 768))
        
        self.renderer = renderer = CuTracer()

        self.viewControl.course = 45
        self.viewControl.pitch = 20


        texFrag = genericFP('tex2D(s_tex, tc0.xy)')

        def resize(x, y):
            print x, y

        def display():
            clearGLBuffers()
                
            with self.viewControl.with_vp:
                 tex = self.renderer.render()
            with ctx(self.viewControl.vp, ortho01, texFrag(s_tex = tex)):
                 drawQuad()

        self.display = display
        self.resize = resize


if __name__ == "__main__":
    import pycuda.autoinit
    App().run()
    
