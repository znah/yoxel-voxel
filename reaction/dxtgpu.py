from __future__ import with_statement
from zgl import *


class DXT1Compressor:
    def __init__(self, size):
        assert (size[0] % 4 == 0) and (size[1] % 4 == 0)
        self.size = size
        self.dxtBuf= RenderTexture(
          size = size, 
          format = GL_LUMINANCE_ALPHA32UI_EXT, 
          srcFormat = GL_LUMINANCE_ALPHA_INTEGER_EXT,
          srcType = GL_INT)

        self.pbo = glGenBuffers(1)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pbo)
        tmpBuf = zeros( (size[1], size[0], 8), uint8 )
        glBufferData(GL_PIXEL_PACK_BUFFER, tmpBuf, GL_STREAM_COPY)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)

        self.dxt1Frag = CGShader("gp4fp", fileName = "compress_YCoCgDXT.cg", entry = "compress_DXT1_fp")

    def compress(self, srcTexture):
        pass

if __name__ == '__main__':
    zglInit((100, 100), "dxt_test")

    compessor = DXT1Compressor((512, 512))




