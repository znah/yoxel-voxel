from __future__ import with_statement
from numpy import *
from zgl import *
from PIL import Image
import time

class TileProvider:
    def __init__(self, fn):
        self.tex = Texture2D(Image.open(fn))
        self.tex.genMipmaps()
        self.tex.setParams( *Texture2D.MipmapLinear )
        #self.tex.setParams( (GL_TEXTURE_MAX_ANISOTROPY_EXT, 16))

        self.texFrag = CGShader("fp40", '''
          uniform sampler2D tex;
          float4 main(float2 texCoord: TEXCOORD0, float4 col : COLOR0) : COLOR
          {
            return tex2D(tex, texCoord) * col;
          }
        ''')
        self.texFrag.tex = self.tex

    def render(self, rect, scale):
        with self.texFrag:
            col = (1, 1, 1)
            #col = random.rand(3)*0.5 + 0.5
            glColor(*col)
            drawQuad()

class VirtualTexture:
    def __init__(self, provider, tileSize, indexSize, cacheSize, border = 1):
        self.provider = provider
        self.tileSize = tileSize
        self.indexSize = indexSize
        self.cacheSize = cacheSize
        self.border = border
        self.padTileSize = tileSize + border * 2

        self.cacheTex = Texture2D(shape = V(cacheSize, cacheSize)*self.padTileSize)
        self.cacheTex.setParams(*Texture2D.Linear)
        self.cacheTex.setParams( (GL_TEXTURE_MAX_ANISOTROPY_EXT, 16))
        self.tileBuf = RenderTexture(shape = (self.padTileSize, self.padTileSize))

        self.lodNum = int(log2(indexSize)) + 1
        lodSizes = [indexSize / 2**lod for lod in xrange(self.lodNum)]
        self.index = [zeros((sz, sz, 3), float32) for sz in lodSizes]

        self.loadTile(self.lodNum-1, (0, 0), (0, 0))
        self.loadTile(3, (1, 1), (1, 0))
        self.loadTile(2, (2, 2), (2, 0))
        self.loadTile(1, (4, 4), (3, 0))
        self.loadTile(0, (8, 8), (4, 3))
        self.loadTile(0, (8, 9), (4, 1))

        self.indexTex = Texture2D(self.index[0], format = GL_RGBA_FLOAT16_ATI)
        with self.indexTex:
            for lod in xrange(1, self.lodNum):
                src = self.index[lod]
                glTexImage2D(GL_TEXTURE_2D, lod, GL_RGBA_FLOAT16_ATI, src.shape[1], src.shape[0], 0, GL_RGB, GL_FLOAT, src)
        self.indexTex.setParams((GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST))
        
        
    def loadTile(self, lod, lodTileIdx, cachePos):
        tileRect = V(lodTileIdx[0], lodTileIdx[1], lodTileIdx[0]+1, lodTileIdx[1]+1)
        scale = 2**lod
        idxRect = tileRect*scale
        texRect = idxRect / self.indexSize
        texBorder = scale * float(self.border)/ (self.tileSize * self.indexSize)
        padTexRect = texRect.copy()
        padTexRect[:2] -= texBorder
        padTexRect[-2:] += texBorder

        for i in xrange(lod + 1):
            self.index[i][idxRect[1]:idxRect[3], idxRect[0]:idxRect[2]] = (cachePos[0], cachePos[1], scale)
            idxRect /= 2

        with ctx(self.tileBuf, Ortho(padTexRect)):
            glClear(GL_COLOR_BUFFER_BIT)
            self.provider.render(padTexRect, scale)
            with self.cacheTex:
                cp = V(*cachePos) * self.padTileSize
                glCopyTexSubImage2D(GL_TEXTURE_2D, 0, cp[0], cp[1], 0, 0, self.padTileSize, self.padTileSize)
    
    def setupShader(self, shader):
        shader.indexTex = self.indexTex
        shader.cacheTex = self.cacheTex
        shader.tileSize = self.tileSize
        shader.indexSize = self.indexSize
        shader.cacheSize = self.cacheSize
        shader.vtexSize = self.indexSize * self.tileSize
        shader.maxLod = self.lodNum - 1
        shader.border = self.border



class App:
    def __init__(self):
        self.viewControl = FlyCamera()
        self.viewControl.speed = 50
        self.viewControl.eye = (0, 0, 10)

        self.tileProvider = TileProvider("bluemarble-east-4096.png")
        self.virtualTex = VirtualTexture(self.tileProvider, 256, 16, 8, border = 4)
        
        self.texFrag = CGShader("fp40", '''
          uniform sampler2D tex;
          float4 main(float2 texCoord: TEXCOORD0) : COLOR
          {
            return tex2D(tex, texCoord);
          }
                  
        ''')
        self.texFrag.tex = self.virtualTex.cacheTex

        self.vtexFrag = CGShader("fp40", '''
          uniform sampler2D indexTex;
          uniform sampler2D cacheTex;
          uniform float tileSize;
          uniform float indexSize;
          uniform float cacheSize;
          uniform float vtexSize;
          uniform float maxLod;
          uniform float border;

          const float eps = 0.00001;

          uniform sampler2D lodColorTex;
          float3 lodCol(float lod) 
          { 
            return tex2D(lodColorTex, float2(lod / 8.0, 0));
          }

          float calcLod(float2 dx, float2 dy)
          {
            float2 d = sqrt(dx*dx + dy * dy);
            float md = max(d.x, d.y) * vtexSize;
            float lod = log2(md)-2;
            return clamp(lod, 0, maxLod-eps);
          }
          
          float4 vtexFetch(float2 texCoord, float lod, float2 dx, float2 dy)
          {
            float3 tileData = tex2Dlod(indexTex, float4(texCoord, 0, lod)).xyz;
            float2 tileIdx = tileData.xy;
            float2 tileScale = tileData.z;
            float2 posInTile = frac(texCoord * indexSize / tileScale);

            float padTileSize = tileSize + 2*border;
            float cacheTexSize = cacheSize * padTileSize;
            float2 posInCache = (tileIdx * padTileSize + border + posInTile*tileSize) / cacheTexSize;

            float dcoef = vtexSize / cacheSize / tileScale;
            return tex2D(cacheTex, posInCache, dx*dcoef, dy*dcoef);
          }

          float4 main(float2 texCoord: TEXCOORD0) : COLOR
          {
            float2 dx = ddx(texCoord);
            float2 dy = ddy(texCoord);
            float lod = calcLod(ddx(texCoord), ddy(texCoord));
            float lodBlend = frac(lod);
            float hiLod = lod - lodBlend;
            float loLod = hiLod + 1;

            float4 c1 = vtexFetch(texCoord, hiLod, dx, dy);
            float4 c2 = vtexFetch(texCoord, loLod, dx, dy);
            float4 c = lerp(c1, c2, lodBlend);
            //c.xyz *= lodCol(lod);
            return c;
          }
                  
        ''')

        lodColors = (1.0 - 0.5 * array([i for i in ndindex(2, 2, 2)])).astype(float32)[newaxis]
        self.lodColorTex = Texture2D(lodColors)
        self.lodColorTex.setParams(*Texture2D.Linear)
        self.vtexFrag.lodColorTex = self.lodColorTex
        
        self.virtualTex.setupShader(self.vtexFrag)

        self.t = time.clock()

    def resize(self, x, y):
        self.viewControl.resize(x, y)

    def idle(self):
        glutPostRedisplay()
    
    def display(self):
        t = time.clock()
        dt = t - self.t;
        self.t = t
        self.viewControl.updatePos(dt)
        
        glViewport(0, 0, self.viewControl.viewSize[0], self.viewControl.viewSize[1])

        with self.viewControl:
            glClearColor(0, 0, 0, 0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            v  = [(0, 0, 0), (100, 0, 0), (100, 100, 0), (0, 100, 0)]
            tc = [(0, 0), (1, 0), (1, 1), (0, 1)]
            with self.vtexFrag:
                drawVerts(GL_QUADS, v, tc)
            with self.texFrag:
                glTranslate(100, 0, 0)
                drawVerts(GL_QUADS, v, tc)
            

        glutSwapBuffers()

    def keyDown(self, key, x, y):
        if ord(key) == 27:
            glutLeaveMainLoop()

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
  
