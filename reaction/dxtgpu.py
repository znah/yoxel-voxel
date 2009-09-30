from __future__ import with_statement
from zgl import *
from PIL import Image
import pylab
from time import clock

class DXT1Compressor:
    def __init__(self, size):
        assert (size[0] % 4 == 0) and (size[1] % 4 == 0)
        self.size = size
        self.dxtSize = V(size)/4
        self.dxtBuf = RenderTexture(
          size = self.dxtSize, 
          format = GL_LUMINANCE_ALPHA32UI_EXT, 
          srcFormat = GL_LUMINANCE_ALPHA_INTEGER_EXT,
          srcType = GL_INT)

        self.resultSize = int(prod(self.dxtSize) * 8)
        self.resultPBO = BufferObject()
        with self.resultPBO.pixelPack:
            glBufferData(GL_PIXEL_PACK_BUFFER, zeros((self.resultSize,), uint8), GL_STREAM_COPY)

        self.dxt1Frag = CGShader("gp4fp", fileName = "compress_YCoCgDXT.cg", entry = "compress_DXT1_fp")

    def compress(self, srcTexture):
        self.dxt1Frag.image = srcTexture
        self.dxt1Frag.imageSize = self.size
        with ctx(self.dxtBuf, ortho, self.dxt1Frag, self.resultPBO.pixelPack):
            drawQuad()
            OpenGL.raw.GL.glReadPixels(0, 0, self.dxtSize[0], self.dxtSize[1], GL_LUMINANCE_ALPHA_INTEGER_EXT, GL_UNSIGNED_INT, None)


if __name__ == '__main__':
    zglInit((100, 100), "dxt_test")

    srcImg = asarray(Image.open("texture.png"))
    src = Texture2D(srcImg)
    print src.size
    compressor = DXT1Compressor(src.size)
    dst = Texture2D(size = src.size, format = GL_COMPRESSED_RGB_S3TC_DXT1_EXT)
    
    
    compressor.compress(src)   # warm up

    glFinish()
    t = clock()
    for i in xrange(100):
        compressor.compress(src)
    glFinish()
    dt = clock() - t
    print dt * 1000

    with ctx(dst, compressor.resultPBO.pixelUnpack):
        glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, src.size[0], src.size[1], GL_COMPRESSED_RGB_S3TC_DXT1_EXT, compressor.resultSize, None)
        unpackedImg = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, 'array')

    Image.fromarray(unpackedImg).save("result.bmp")
    
