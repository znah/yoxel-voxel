from __future__ import with_statement
from numpy import *
from zgl import *
from PIL import Image
import time

class TileProvider:
    def __init__(self, fn, tileSize, indexSize, tileBorder = 1):
        self.tileSize = tileSize
        self.indexSize = indexSize
        self.tileBorder = tileBorder
        self.vtexSize = tileSize * indexSize
        self.padTileSize = tileSize + tileBorder*2
        
        self.tex = Texture2D(Image.open(fn))
        self.tex.genMipmaps()
        self.tex.setParams( *Texture2D.MipmapLinear )
        self.texFrag = CGShader("fp40", '''
          uniform sampler2D tex;
          float4 main(float2 texCoord: TEXCOORD0, float4 col : COLOR0) : COLOR
          {
            return tex2D(tex, texCoord);// * col;
          }
        ''')
        self.texFrag.tex = self.tex

    def render(self, lod, tileIdx):
        scale = 2.0**lod
        tileTexSize = self.tileSize * scale / self.vtexSize
        borderTexSize = self.tileBorder * scale / self.vtexSize
        (x1, y1) = V(tileIdx) * tileTexSize - borderTexSize
        (x2, y2) = (V(tileIdx)+1) * tileTexSize + borderTexSize
        
        with Ortho((x1, y1, x2, y2)):
            col = random.rand(3)*0.5 + 0.5
            glColor(*col)
            with self.texFrag:
                drawQuad()

class VirtualTexture:
    def __init__(self, provider, cacheSize):
        self.provider = provider
        self.padTileSize = provider.padTileSize
        self.indexSize = provider.indexSize
        self.cacheSize = cacheSize
        self.cacheTexSize = cacheSize * self.padTileSize

        self.cacheTex = Texture2D(shape = (self.cacheTexSize, self.cacheTexSize))
        self.cacheTex.setParams(*Texture2D.Linear)
        self.cacheTex.setParams( (GL_TEXTURE_MAX_ANISOTROPY_EXT, 16))
        self.tileBuf = RenderTexture(shape = (self.padTileSize, self.padTileSize))

        self.lodNum = int(log2(self.indexSize)) + 1
        lodSizes = [self.indexSize / 2**lod for lod in xrange(self.lodNum)]
        self.index = [zeros((sz, sz, 3), float32) for sz in lodSizes]

        idx = 0
        for lod in xrange(0, self.lodNum):
            lodTileNum = self.indexSize / 2**lod
            for i in xrange(lodTileNum):
                self.loadTile(lod, (i, i/2), idx)
                idx += 1

        self.indexTex = Texture2D(shape = (self.indexSize, self.indexSize), format = GL_RGBA_FLOAT16_ATI)
        with self.indexTex:
            for lod in xrange(0, self.lodNum):
                src = self.index[lod]
                glTexImage2D(GL_TEXTURE_2D, lod, GL_RGBA_FLOAT16_ATI, src.shape[1], src.shape[0], 0, GL_RGB, GL_FLOAT, src)
        self.indexTex.setParams((GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST))
        
        
    def loadTile(self, lod, lodTileIdx, cacheIdx):
        cachePos = unravel_index(cacheIdx, (self.cacheSize, self.cacheSize))
        scale = 2**lod
        lo = V(lodTileIdx) * scale
        hi = lo + scale
        for i in xrange(lod+1):
            rng = self.index[i][ lo[1]:hi[1], lo[0]:hi[0] ]
            mark = (rng[:,:,2] > scale) | (rng[:,:,2] == 0)
            rng[mark] = (cachePos[0], cachePos[1], scale)
            lo /= 2
            hi /= 2

        with ctx(self.tileBuf):
            glClear(GL_COLOR_BUFFER_BIT)
            self.provider.render(lod, lodTileIdx)
            with self.cacheTex:
                cp = V(*cachePos) * self.padTileSize
                glCopyTexSubImage2D(GL_TEXTURE_2D, 0, cp[0], cp[1], 0, 0, self.padTileSize, self.padTileSize)
    
    def setupShader(self, shader):
        shader.indexTex = self.indexTex
        shader.cacheTex = self.cacheTex
        shader.tileSize = self.provider.tileSize
        shader.indexSize = self.indexSize
        shader.vtexSize = self.provider.vtexSize
        shader.maxLod = self.lodNum - 1
        shader.border = self.provider.tileBorder
        shader.padTileSize = self.padTileSize
        shader.cacheTexSize = self.cacheTexSize



class App:
    def __init__(self):
        self.viewControl = FlyCamera()
        self.viewControl.speed = 50
        self.viewControl.eye = (0, 0, 10)

        self.tileProvider = TileProvider("img/track_1_2.jpg", 256, 16, 8)
        self.virtualTex = VirtualTexture(self.tileProvider, 8)
        
        self.texFrag = CGShader("fp40", '''
          uniform sampler2D tex;
          float4 main(float2 texCoord: TEXCOORD0) : COLOR
          {
            return tex2D(tex, texCoord);
          }
                  
        ''')
        self.texFrag.tex = self.virtualTex.cacheTex

        self.vtexFrag = CGShader("fp40", fileName = 'vtex.cg')
        self.vtexFeedbackFrag = CGShader("fp40", fileName = 'vtexFeedback.cg')
        self.virtualTex.setupShader(self.vtexFrag)
        self.virtualTex.setupShader(self.vtexFeedbackFrag)

        self.t = time.clock()

    def resize(self, x, y):
        self.viewControl.resize(x, y)

    def idle(self):
        glutPostRedisplay()
    
    def renderGeom(self):
        v  = [(0, 0, 0), (100, 0, 0), (100, 100, 0), (0, 100, 0)]
        tc = [(0, 0), (1, 0), (1, 1), (0, 1)]
        with self.viewControl:
            drawVerts(GL_QUADS, v, tc)

    def display(self):
        t = time.clock()
        dt = t - self.t;
        self.t = t
        self.viewControl.updatePos(dt)
        
        glViewport(0, 0, self.viewControl.viewSize[0], self.viewControl.viewSize[1])
        with self.viewControl:
            glClearColor(0, 0, 0, 0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            with self.vtexFrag:
                self.renderGeom()
            with self.texFrag:
                glTranslate(110, 0, 0)
                glScale(100, 100, 1)
                drawQuad()
        glutSwapBuffers()

    def fetchFeedback(self): 
        pass

    def keyDown(self, key, x, y):
        if ord(key) == 27:
            glutLeaveMainLoop()
        elif key == '1':
            self.fetchFeedback()
        else:
            self.viewControl.keyDown(key, x, y)
               
    def keyUp(self, key, x, y):
        self.viewControl.keyUp(key, x, y)

    def mouseMove(self, x, y):
        self.viewControl.mouseMove(x, y)

    def mouseButton(self, btn, up, x, y):
        self.viewControl.mouseButton(btn, up, x, y)




if __name__ == "__main__":
  glutInit([])
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
  glutInitWindowSize(800, 600)
  glutCreateWindow("hello")
  InitCG()

  app = App()
  glutSetCallbacks(app)

  #wglSwapIntervalEXT(0)
  glutSetOption ( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION )
  glutMainLoop()
  
