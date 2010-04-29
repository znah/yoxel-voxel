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


class CuTracer(HasTraits):
    def __init__(self):
        mod = cu_compile( file('trace.cu').read() )
        Trace        = mod.get_function('Trace')

        c_viewSize    = mod.get_global('c_viewSize')
        c_proj2wldMtx = mod.get_global('c_proj2wldMtx')
        c_eyePos      = mod.get_global('c_eyePos')

        w, h = 800, 600
        self.a = a = zeros((h, w, 4), uint8) + 128
        viewTex = Texture2D(img = a)
        viewBuf  = CuGLBuf(a)

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
            Trace(int32(mapping.device_ptr()), block = (32, 8, 1), grid = (w/32, h/8))  # mapping.size()
            mapping.unmap()
            with ctx(viewTex, viewBuf.gl.pixelUnpack):
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, None)
            return viewTex
        self.render = render


class App(ZglAppWX):
    renderer = Instance(CuTracer)    
    
    def __init__(self):
        ZglAppWX.__init__(self, viewControl = FlyCamera(), zglpath='..')
        
        self.renderer = renderer = CuTracer()

        texFrag = genericFP('tex2D(s_tex, tc0.xy)')

        def display():
            clearGLBuffers()
                
            with self.viewControl.with_vp:
                 tex = self.renderer.render()
            with ctx(self.viewControl.vp, ortho, texFrag(s_tex = tex)):
                 drawQuad()

        self.display = display


if __name__ == "__main__":
    import pycuda.autoinit
    App().run()
    
