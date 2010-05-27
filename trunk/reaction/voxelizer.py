from __future__ import with_statement
from zgl import *
from volvis import VolumeRenderer

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
        a[i] = ((bits>>i) & 1) * 255
    return a


class Voxelizer:
    def __init__(self, size = 256):
        self.size = size
        self.sliceNum = sliceNum = size / 128
        assert sliceNum <= 8

        self.slices = TextureArray(size = (size, size, sliceNum),
              format = GL_RGBA32UI_EXT,
              srcFormat = GL_RGBA_INTEGER_EXT,
              srcType = GL_UNSIGNED_INT)
          
        self.fragProg = CGShader('gp4fp', '''
          #line 36
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
            tc = mul(glstate.matrix.modelview[0], pos);
          }
        ''')
        
        self.fbo = Framebuffer()
        with self.fbo:
            for i in xrange(sliceNum):
                glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, self.slices, 0, i)
            glDrawBuffers(sliceNum, GL_COLOR_ATTACHMENT0+arange(sliceNum))

        self.vpCtx = ctx(Viewport(0, 0, self.size, self.size), ortho)

        self.clear()

    def clear(self):
        with self:
            clearGLBuffers()
        
    def __enter__(self):
        self.state = ctx(
          self.fbo, self.vpCtx,
          self.vertProg, self.fragProg,
          glstate(GL_COLOR_LOGIC_OP, GL_DEPTH_CLAMP_NV)
        )
        self.state.__enter__()
        glLogicOp(GL_XOR)
        
    def __exit__(self, *args):
        self.state.__exit__(*args)
        del self.state

    def dump(self):
        bits = zeros( (self.sliceNum, self.size, self.size, 4), uint32 )
        with self.slices:
            OpenGL.raw.GL.glGetTexImage(GL_TEXTURE_2D_ARRAY_EXT, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT, bits.ctypes.data )
        res = zeros((self.size,)*3, uint8)
        for i in xrange(self.sliceNum):
            for j in xrange(4):
                dst = res[i * 128 + j*32:][:32]
                src = expand_uint32(bits[i,:,:,j])
                dst[:] = src
        return res

    def dumpToPBO(self):
        if not hasattr(self, "dumpPBO"):
            bits = zeros( (self.sliceNum, self.size, self.size, 4), uint32 )
            self.dumpPBO = BufferObject(data = bits, use = GL_STREAM_COPY)
        with ctx(self.dumpPBO.pixelPack, self.slices):
            OpenGL.raw.GL.glGetTexImage(GL_TEXTURE_2D_ARRAY_EXT, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT, None)
        return self.dumpPBO

    @with_(glprofile('dumpToTex'))
    def dumpToTex(self):
        if not hasattr(self, "dumpTex"):
            self.dumpTex = Texture3D(size = (self.size,)*3, format = GL_LUMINANCE8)
            self.dumpFBO = Framebuffer()
            with self.dumpFBO:
                glDrawBuffers(4, GL_COLOR_ATTACHMENT0+arange(4))
            self.dumpProg = CGShader('gp4fp', '''
              typedef unsigned int uint;

              uniform usampler2DARRAY slices;
              uniform float slice;
              uniform int mask;
              
              void main(float2 pos: TEXCOORD0,
                out float layer0: COLOR0,
                out float layer1: COLOR1,
                out float layer2: COLOR2,
                out float layer3: COLOR3
              )
              {
                uint4 bits = tex2DARRAY(slices, float3(pos, slice));
                layer0 = (bits.r & mask) ? 1.0f : 0.0f;
                layer1 = (bits.g & mask) ? 1.0f : 0.0f;
                layer2 = (bits.b & mask) ? 1.0f : 0.0f;
                layer3 = (bits.a & mask) ? 1.0f : 0.0f;
              }
            ''')
            self.dumpProg.slices = self.slices

        with ctx(self.dumpFBO, self.vpCtx, self.dumpProg):
            for sl in xrange(self.sliceNum):
                self.dumpProg.slice = sl
                for bit in xrange(32):
                    self.dumpProg.mask = 1<<bit
                    layer = sl * 128 + bit
                    for ch in xrange(4):
                        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + ch, self.dumpTex, 0, layer + ch*32)
                    drawQuad()
        return self.dumpTex


    @with_(glprofile('densityToTex'))
    def densityToTex(self):
        halfSize = self.size / 2
        densityTex = Texture3D(size = (halfSize,)*3, format = GL_LUMINANCE8)
        fbo = Framebuffer()
        with fbo:
            glDrawBuffers(8, GL_COLOR_ATTACHMENT0+arange(8))
        densityFrag = CGShader('gp4fp', '''
          #line 200
          typedef unsigned int uint;

          uniform usampler2DARRAY slices;
          uniform float dx;
          uniform float slice;
          uniform int channel;
          uniform float oddFlag;
          

          uint fetchChannel(float2 pos)
          {
            uint4 d = tex2DARRAY(slices, float3(pos, slice));
            uint res = d.x;
            if (channel == 1)
              res = d.y;
            if (channel == 2)
              res = d.z;
            if (channel == 3)
              res = d.w;
            return res;
          }

          uint sumPairs(uint v)
          {
            const uint mask = 0x55555555;
            return ((v>>1)&mask) + (v&mask);
          }

          float bits2float(uint v)
          {
            return float(v & 0xF) * (1.0 / 8.0);
          }

          void main(float2 pos: TEXCOORD0,
            out float layer0: COLOR0,
            out float layer1: COLOR1,
            out float layer2: COLOR2,
            out float layer3: COLOR3,
            out float layer4: COLOR4,
            out float layer5: COLOR5,
            out float layer6: COLOR6,
            out float layer7: COLOR7
          )
          {
            uint v00 = fetchChannel(pos + float2(-dx, -dx));
            uint v01 = fetchChannel(pos + float2(-dx,  dx));
            uint v10 = fetchChannel(pos + float2( dx, -dx));
            uint v11 = fetchChannel(pos + float2( dx,  dx));
            v00 = sumPairs(v00);
            v01 = sumPairs(v01);
            v10 = sumPairs(v10);
            v11 = sumPairs(v11);

            const uint mask2 = 0x33333333;
            uint veven = (v00 & mask2) + (v01 & mask2) + (v10 & mask2) + (v11 & mask2);
            uint vodd  = ((v00>>2) & mask2) + ((v01>>2) & mask2) + ((v10>>2) & mask2) + ((v11>>2) & mask2);

            uint res = oddFlag ? vodd : veven;

            layer0 = bits2float(res);
            layer1 = bits2float(res >> 4);
            layer2 = bits2float(res >> 8);
            layer3 = bits2float(res >> 12);
            layer4 = bits2float(res >> 16);
            layer5 = bits2float(res >> 20);
            layer6 = bits2float(res >> 24);
            layer7 = bits2float(res >> 28);
          }
        ''')
        densityFrag.slices = self.slices
        densityFrag.dx = 0.5 / self.size

        viewCtx = ctx(Viewport(0, 0, halfSize, halfSize), ortho)

        import itertools as it

        @with_(glprofile("densityToTex"))
        def newDensityToTex():
            with ctx(fbo, viewCtx, densityFrag):
                for sl in xrange(self.sliceNum):
                    densityFrag.slice = sl
                    for ch, odd in it.product( xrange(4), (0, 1) ):
                        densityFrag.channel = ch
                        densityFrag.oddFlag = odd
                        for i in xrange(8):
                            layer = sl*64 + ch*16 + i*2 + odd
                            glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, densityTex, 0, layer)
                        drawQuad()
            return densityTex
        self.densityToTex = newDensityToTex
        return newDensityToTex()
            


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
    multiSample  = Bool(True)
    turn         = Range(0.0, 360.0, 0.0, mode='slider')

    @on_trait_change('multiSample')
    def selectVoxelizer(self):
        if not self.multiSample:
            self.voxelizer    = self.simpleVox
            self.voxelTexFunc = lambda : self.simpleVox.dumpToTex()
        else:
            self.voxelizer    = self.multiVox
            self.voxelTexFunc = lambda : self.multiVox.densityToTex()

    def __init__(self):
        ZglAppWX.__init__(self, viewControl = FlyCamera())

        size = 512
        self.simpleVox = Voxelizer(size)
        self.multiVox = Voxelizer(size*2)
        self.selectVoxelizer()
        
        (v, f) = load_obj("data/bunny/bunny.obj") #"data/bunny/bunny.obj"
        v = fit_box(v)[:,[0, 2, 1]]

        self.vertBuf = BufferObject(v)
        self.idxBuf = BufferObject(f)
        self.idxNum = len(f) * 3

        self.volumeRender = VolumeRenderer()
        
    def drawMesh(self):
        with self.vertBuf.array:
            glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, None)
        with ctx(self.idxBuf.elementArray, vattr(0)):
            glDrawElements(GL_TRIANGLES, self.idxNum, GL_UNSIGNED_INT, None)
    def drawFrame(self):
        d = 1.0 / self.voxelizer.size
        drawBox((0, 0, 0), (1, 1, 1))
        drawBox((d, d, 0), (1-d, 1-d, 1))
        drawBox((d, 0, d), (1-d, 1, 1-d))
        drawBox((0, d, d), (1, 1-d, 1-d))

    def display(self):
        clearGLBuffers()
        
        with ctx(glprofile('voxelize'), self.voxelizer):
            clearGLBuffers()
            #self.drawFrame()
            glTranslate(0.5, 0.5, 0)
            glRotate(self.turn, 0, 0, 1)            
            glTranslate(-0.5, -0.5, 0)
            self.drawMesh()
        self.volumeRender.volumeTex = self.voxelTexFunc()

        with ctx(glprofile('volumeRender'), self.viewControl.with_vp):
            self.volumeRender.render()

if __name__ == "__main__":
    App().run()
