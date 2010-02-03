from __future__ import with_statement
from zgl import *
from OpenGL.GL.NV.geometry_program4 import *
from volvis import VolumeRenderer

from StringIO import StringIO

def load_obj(fn):
    ss = file(fn).readlines()
    vs = [s[1:] for s in ss if s[0] == 'v']
    fs = [s[1:] for s in ss if s[0] == 'f']
    verts = loadtxt( StringIO("".join(vs)), float32 )
    faces = loadtxt( StringIO("".join(fs)), int32 ) - 1
    return (verts, faces)

def fit_box(v):
    lo = v.min(0)
    hi = v.max(0)
    d = hi - lo
    scale = 1.0 / d.max()
    v = v - lo
    v *= scale
    v += (0.5, 0.5, 0.5) - 0.5*d*scale
    return v


def expand_uint32(bits):
    a = zeros( (32,) + bits.shape[:2], uint8 )
    for i in xrange(32):
        a[i] = ((bits>>i) & 1) * 128#255
    return a


class Voxelizer:
    def __init__(self, size = 256):
        self.size = size
        self.sliceNum = sliceNum = size / 128
        assert sliceNum <= 8

        # TODO: texture array
        self.slices = []
        for i in xrange(sliceNum):
            tex = Texture2D(
              size = (size, size), 
              format = GL_RGBA32UI_EXT,
              srcFormat = GL_RGBA_INTEGER_EXT,
              srcType = GL_UNSIGNED_INT)
            self.slices.append(tex)  
          
        self.fragProg = CGShader('gp4fp', '''
          #line 38
          uniform usampler1D columnBits;
          uniform float sliceNum;
          
          struct PixelOutput
          {
            uint4 s0 : COLOR0;
            uint4 s1 : COLOR1;
            uint4 s2 : COLOR2;
            uint4 s3 : COLOR3;
            uint4 s4 : COLOR4;
            uint4 s5 : COLOR5;
            uint4 s6 : COLOR6;
            uint4 s7 : COLOR7;
          };

          PixelOutput main(float3 pos : TEXCOORD0)
          {
            PixelOutput output;
            uint4 zeros = uint4(0);
            uint4 ones  = uint4(0xffffffff);
            
            const float sliceDepth = 128.0f;
            const float halfVoxel = 0.5f / sliceDepth;
            float z = pos.z * sliceNum - halfVoxel;    // TODO: use WPOS (no need vertProg)
            uint4 bits = tex1D( columnBits, frac(z) );
            
            output.s0 = (z < 0.0f) ? zeros : (z < 1.0f ? bits : ones);
            output.s1 = (z < 1.0f) ? zeros : (z < 2.0f ? bits : ones);
            output.s2 = (z < 2.0f) ? zeros : (z < 3.0f ? bits : ones);
            output.s3 = (z < 3.0f) ? zeros : (z < 4.0f ? bits : ones);
            output.s4 = (z < 4.0f) ? zeros : (z < 5.0f ? bits : ones);
            output.s5 = (z < 5.0f) ? zeros : (z < 6.0f ? bits : ones);
            output.s6 = (z < 6.0f) ? zeros : (z < 7.0f ? bits : ones);
            output.s7 = (z < 7.0f) ? zeros : (z < 8.0f ? bits : ones);
            return output;
          }
        ''')
        self.fragProg.sliceNum = sliceNum
        
        bits32 = array([(1<<i) - 1 for i in xrange(1, 33)], uint32)
        bits = zeros( (128, 4), uint32 )
        bits[ 32:,0] = 0xffffffff
        bits[ 64:,1] = 0xffffffff
        bits[ 96:,2] = 0xffffffff
        bits[  0: 32, 0] = bits32
        bits[ 32: 64, 1] = bits32
        bits[ 64: 96, 2] = bits32
        bits[ 96:128, 3] = bits32
        columnBits = Texture1D(
          img = bits, 
          format = GL_RGBA32UI_EXT, 
          srcType = GL_UNSIGNED_INT,
          srcFormat = GL_RGBA_INTEGER_EXT)
        self.fragProg.columnBits = columnBits  
          
        self.vertProg = CGShader('vp40', '''
          void main( 
            float4 pos : ATTR0,
            out float4 oPos : POSITION,
            out float4 tc   : TEXCOORD0) 
          { 
            oPos = mul(glstate.matrix.mvp, pos);
            tc = pos;
          }
        ''')
        
        self.fbo = Framebuffer()
        with self.fbo:
            glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, self.fbo)
            for i in xrange(sliceNum):
                glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT + i, GL_TEXTURE_2D, self.slices[i], 0)
            glDrawBuffers(sliceNum, GL_COLOR_ATTACHMENT0_EXT+arange(sliceNum))

        self.clear()

    def clear(self):
        with self:
            clearGLBuffers()
        
    def __enter__(self):
        self.state = ctx(
          self.fbo, Viewport(0, 0, self.size, self.size), ortho,
          self.vertProg, self.fragProg,
          glstate(GL_COLOR_LOGIC_OP, GL_DEPTH_CLAMP_NV)
        )
        self.state.__enter__()
        glLogicOp(GL_XOR)
        
    def __exit__(self, *args):
        self.state.__exit__(*args)
        del self.state

    def dump(self):
        res = zeros((self.size,)*3, uint8)
        for i, sliceTex in enumerate(self.slices):
            with sliceTex:
                a = zeros( (self.size, self.size, 4), uint32 )
                OpenGL.raw.GL.glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT, a.ctypes.data )
            for j in xrange(4):
                dst = res[i * 128 + j*32:][:32]
                src = expand_uint32(a[...,j])
                dst[:] = src
        return res


def drawBox(lo = (0, 0, 0), hi = (1, 1, 1)):
    i = indices((2, 2, 2)).T.reshape(-1,3)
    verts = choose(i, (lo, hi)).astype(float32)
    idxs = [ 0, 2, 3, 1,
             0, 1, 5, 4, 
             4, 5, 7, 6,
             1, 3, 7, 5,
             0, 4, 6, 2,
             2, 6, 7, 3]
    glBegin(GL_QUADS)
    for i in idxs:
        glVertex3f(*verts[i])
    glEnd()
    

class App(ZglAppWX):
    volumeRender = Instance(VolumeRenderer)

    def __init__(self):
        ZglAppWX.__init__(self, viewControl = FlyCamera())
        self.fragProg = CGShader('fp40', TestShaders, entry = 'TexCoordFP')
        
        self.voxelizer = Voxelizer(256)
        
        (v, f) = load_obj("data/bunny/bunny.obj")
        v = fit_box(v)
        z = v[:,2].copy()
        v[:,2] = v[:,1]
        v[:,1] = z

        self.vertBuf = BufferObject(v)
        self.idxBuf = BufferObject(f)
        self.idxNum = len(f) * 3

        with self.voxelizer:
            with self.vertBuf.array:
                glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, None)
            with ctx(self.idxBuf.elementArray, vattr(0)):
                glDrawElements(GL_TRIANGLES, self.idxNum, GL_UNSIGNED_INT, None)
                
            d = 1.0 / 256
            drawBox((0, 0, 0), (1, 1, 1))
            drawBox((d, d, 0), (1-d, 1-d, 1))
            drawBox((d, 0, d), (1-d, 1, 1-d))
            drawBox((0, d, d), (1, 1-d, 1-d))
        
        a = self.voxelizer.dump()
        self.volumeRender = VolumeRenderer(Texture3D(img=a))

    def display(self):
        clearGLBuffers()
        with ctx(self.viewControl.with_vp):
            self.volumeRender.render()

if __name__ == "__main__":
    App().run()
