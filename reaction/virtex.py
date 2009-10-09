from __future__ import with_statement
from numpy import *
from zgl import *
from PIL import Image
from time import clock
from scipy import weave
import vtex

class TileProvider:
    def __init__(self, texFile, tileSize, indexSize, tileBorder = 1):
        self.tileSize = tileSize
        self.indexSize = indexSize
        self.tileBorder = tileBorder
        self.vtexSize = tileSize * indexSize
        self.padTileSize = tileSize + tileBorder*2
        
        self.noiseTex = Texture2D(random.rand(512, 512, 4).astype(float32))
        self.noiseTex.genMipmaps()
        self.noiseTex.setParams( *Texture2D.MipmapLinear )

        self.tex = Texture2D(Image.open(texFile))
        self.tex.genMipmaps()
        self.tex.setParams( *Texture2D.MipmapLinear )

        self.texFrag = CGShader("fp40", '''
          uniform sampler2D tex;
          uniform sampler2D noiseTex;
          uniform float noiseScale;

          float perlin(float2 pos)
          {
            float v = 0;
            float amp = 1.0;
            for (int i = 0; i < 9; ++i)
            {
              v += amp * abs(tex2D(noiseTex, pos).r * 2 - 1);
              amp /= 1.3;
              pos *= 2;
            }
            return v;
          }

          float4 main(float2 texCoord: TEXCOORD0, float4 col : COLOR0) : COLOR
          {
            float4 c = tex2D(tex, texCoord);
            float v = perlin(texCoord*noiseScale / 128)*0.5;
            c.rgb = 0.7*c.rgb + float3(0.3 * v);
            return c;
          }
        ''')
        self.texFrag.tex = self.tex
        self.texFrag.noiseTex = self.noiseTex
        self.texFrag.noiseScale = self.vtexSize / 512

    def render(self, lod, tileIdx):
        scale = 2.0**lod
        tileTexSize = self.tileSize * scale / self.vtexSize
        borderTexSize = self.tileBorder * scale / self.vtexSize
        (x1, y1) = V(tileIdx) * tileTexSize - borderTexSize
        (x2, y2) = (V(tileIdx)+1) * tileTexSize + borderTexSize
        
        with Ortho((x1, y1, x2, y2)):
            col = (1, 1, 1)
            #col = random.rand(3)*0.5 + 0.5
            glColor(*col)
            with self.texFrag:
                drawQuad()
        with Ortho((0, 0, self.padTileSize, self.padTileSize)):
            glWindowPos2i(10, 15)
            glutBitmapString(GLUT_BITMAP_9_BY_15, "(%d, %d) %d" % (tileIdx + (lod,)))
            '''
            glBegin(GL_LINE_LOOP)
            a = self.tileBorder + 0.5
            b = self.tileBorder + self.tileSize - 0.5
            glVertex(a, a)
            glVertex(a, b)
            glVertex(b, b)
            glVertex(b, a)
            glEnd()
            '''



def makegrid(x, y):
    a = zeros((y, x, 2), float32)
    a[...,1], a[...,0] = indices((y, x))
    return a


class App:
    def __init__(self, viewSize):
        self.viewControl = FlyCamera()
        self.viewControl.speed = 50
        self.viewControl.eye = (0, 0, 10)
        self.viewControl.zFar = 10000

        self.tileProvider = TileProvider("img/sand.jpg", 512, 512, 8)
        self.virtualTex = vtex.VirtualTexture(self.tileProvider, 10)
        
        self.texFrag = CGShader("fp40", '''
          uniform sampler2D tex;
          float4 main(float2 texCoord: TEXCOORD0) : COLOR
          {
            return tex2D(tex, texCoord);
          }
                  
        ''')
        self.texFrag.tex = self.virtualTex.cacheTex

        self.vtexFrag = CGShader("fp40", fileName = 'vtex.cg')
        self.vtexFeedbackFrag = CGShader("gp4fp", fileName = 'vtex.cg', entry = 'feedback')
        self.virtualTex.setupShader(self.vtexFrag)
        self.virtualTex.setupShader(self.vtexFeedbackFrag)

        self.initTerrain()

        fbSize = (400, 300)
        self.feedbackBuf = RenderTexture(size = fbSize, 
          format = GL_LUMINANCE32UI_EXT, 
          srcFormat = GL_LUMINANCE_INTEGER_EXT,
          srcType = GL_INT,
          depth = True)
        self.feedbackArray = zeros((prod(fbSize),), uint32)

        self.vtexUpdateTime = clock()
        self.moved = True

        self.t = clock()

    def initTerrain(self):
        heightmap = asarray(Image.open("img/heightmap.png"))
        sy, sx = heightmap.shape
        wldStep = 1000.0 / (sx-1)
        texStep = 1.0 / (sx-1)
        grid = makegrid(sx, sy)
        self.terrainVerts = zeros((sy, sx, 3), float32)
        self.terrainVerts[:,:,:2] = grid * wldStep
        self.terrainVerts[:,:,2] = heightmap / 255.0 * 50.0
        self.terrainTexCoords = grid * texStep

        #newX = cos(self.terrainTexCoords[:,:,0]*2*pi) * (200.0 - self.terrainVerts[:,:,2])
        #newZ = sin(self.terrainTexCoords[:,:,0]*2*pi) * (200.0 - self.terrainVerts[:,:,2])
        #self.terrainVerts[:,:,0] = newX
        #self.terrainVerts[:,:,2] = newZ
        
        idxgrid = arange(sx*sy).reshape(sy, sx)
        self.terrainIdxs = zeros((sy-1, sx-1, 4), uint32)
        self.terrainIdxs[...,0] = idxgrid[ :-1, :-1 ]
        self.terrainIdxs[...,1] = idxgrid[ :-1,1:   ]  
        self.terrainIdxs[...,2] = idxgrid[1:  ,1:   ]
        self.terrainIdxs[...,3] = idxgrid[1:  , :-1 ]
        self.terrainIdxs = self.terrainIdxs.flatten()

    def renderTerrain(self):
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.terrainVerts)
        glClientActiveTexture(GL_TEXTURE0)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glTexCoordPointer(2, GL_FLOAT, 0, self.terrainTexCoords)
        glDrawElements(GL_QUADS, len(self.terrainIdxs), GL_UNSIGNED_INT, self.terrainIdxs)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    def resize(self, x, y):
        self.viewControl.resize(x, y)

    def idle(self):
        glutPostRedisplay()
    
    def display(self):
        t = clock()
        dt = t - self.t;
        self.t = t
        self.viewControl.updatePos(dt)

        if self.moved:
            self.updateVTex()
            self.moved = False
        
        glEnable(GL_DEPTH_TEST)
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glViewport(0, 0, self.viewControl.viewSize[0], self.viewControl.viewSize[1])
        with self.viewControl:
            with self.vtexFrag:
                self.renderTerrain()
            with self.texFrag:
                glTranslate(-110, 0, 0)
                glScale(100, 100, 1)
                drawQuad()


        glutSwapBuffers()

    def fetchFeedback(self):
        self.vtexFeedbackFrag.dcoef = V(self.feedbackBuf.size()) / V(self.viewControl.viewSize)
        with ctx(self.feedbackBuf, self.viewControl, self.vtexFeedbackFrag):
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.renderTerrain()
            (sx, sy) = self.feedbackBuf.size()
            OpenGL.raw.GL.glReadPixels(0, 0, sx, sy, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_INT, self.feedbackArray.ctypes.data)
        bits = unique(self.feedbackArray)
        tiles = [((x>>24) - 1, x & 0xfff, (x>>12) & 0xfff) for x in bits if x != 0]
        
        def parent(t):
            return (t[0]+1, t[1]/2, t[2]/2)
        def parents(ts, maxLod):
            return [ parent(t) for t in ts if t[0] != maxLod ]
        result = set([])
        while len(tiles) > 0:
            result.update(tiles)
            tiles = parents(tiles, self.virtualTex.lodNum-1)
        return result        

    def updateVTex(self):
        glFinish()
        t0 = clock()
        tiles = self.fetchFeedback()
        glFinish()
        t1 = clock()
        updated = self.virtualTex.updateCache(tiles)
        glFinish()
        t2 = clock()
        print "tileNum: %d,  feedback: %d ms, udate: %d ms,  %d tiles upd" % (len(tiles), (t1-t0)*1000, (t2-t1)*1000, len(updated))

    def keyDown(self, key, x, y):
        if ord(key) == 27:
            glutLeaveMainLoop()
        elif key == ' ':
            self.updateVTex()
        else:
            self.viewControl.keyDown(key, x, y)
            self.moved = True
                
    def keyUp(self, key, x, y):
        self.viewControl.keyUp(key, x, y)
        self.moved = True

    def mouseMove(self, x, y):
        self.viewControl.mouseMove(x, y)
        self.moved = True

    def mouseButton(self, btn, up, x, y):
        self.viewControl.mouseButton(btn, up, x, y)
        self.moved = True



if __name__ == "__main__":
  viewSize = (800, 600)
  zglInit(viewSize, "hello")

  app = App(viewSize)
  glutSetCallbacks(app)

  #wglSwapIntervalEXT(0)
  glutMainLoop()
  
