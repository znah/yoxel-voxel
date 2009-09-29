from __future__ import with_statement
from numpy import *
from zgl import *
from PIL import Image
from time import clock

class VirtualTexture:
    def __init__(self, provider, cacheSize):
        self.provider = provider
        self.padTileSize = provider.padTileSize
        self.indexSize = provider.indexSize
        self.cacheSize = cacheSize
        self.cacheTexSize = cacheSize * self.padTileSize

        self.cacheTex = Texture2D(size = (self.cacheTexSize, self.cacheTexSize))
        self.cacheTex.setParams(*Texture2D.Linear)
        self.cacheTex.setParams( (GL_TEXTURE_MAX_ANISOTROPY_EXT, 16))
        self.tileBuf = RenderTexture(size = (self.padTileSize, self.padTileSize))

        self.createIndex()
        self.cachedTiles = dict() # tile to cacheIdx
        self.updateCache( [(self.lodNum-1, 0, 0)] )
    
    def createIndex(self):
        self.lodNum = int(log2(self.indexSize)) + 1
        lodSizes = [self.indexSize / 2**lod for lod in xrange(self.lodNum)]
        self.indexTex = Texture2D(size = (self.indexSize, self.indexSize), format = GL_RGBA_FLOAT16_ATI)
        zeroBuf = zeros((self.indexSize**2,3), float32)
        with self.indexTex:
            for lod, size in enumerate(lodSizes):
                glTexImage2D(GL_TEXTURE_2D, lod, GL_RGBA_FLOAT16_ATI, size, size, 0, GL_RGB, GL_FLOAT, zeroBuf)
        self.indexTex.setParams((GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST))

    def renderTile(self, tile, cacheIdx):
        (lod, x, y) = tile
        with ctx(self.tileBuf):
            glClear(GL_COLOR_BUFFER_BIT)
            self.provider.render(lod, (x, y))
            with self.cacheTex:
                cp = V(cacheIdx) * self.padTileSize
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

    def updateCache(self, tiles):
        cacheCapacity = self.cacheSize**2
        if len(tiles) > cacheCapacity:
            print "non enough cache!", len(tiles), cacheCapacity

        tiles = list(tiles)
        tiles.sort(reverse=True)
        tiles = tiles[:cacheCapacity]
        
        visible = set(tiles)
        cached = set(self.cachedTiles.keys())
        disposable = cached - visible
        toInsert = visible - cached

        toRender = []
        with self.indexTex:
            for tile in toInsert:
                if len(self.cachedTiles) < cacheCapacity:
                   cacheIdx = unravel_index(len(self.cachedTiles), (self.cacheSize, self.cacheSize))
                else:
                   oldTile = disposable.pop()
                   cacheIdx = self.cachedTiles[oldTile]
                   self.cachedTiles.pop(oldTile)
                   (lod, x, y) = oldTile
                   glTexSubImage2D(GL_TEXTURE_2D, lod, x, y, 1, 1, GL_RGB, GL_FLOAT, [0, 0, 0])
                (lod, x, y) = tile
                lodSize = self.indexSize / 2**lod
                if (x >= lodSize or y >= lodSize ):
                  print "out!!!"
                  continue
                scale = 2**lod
                glTexSubImage2D(GL_TEXTURE_2D, lod, x, y, 1, 1, GL_RGB, GL_FLOAT, [cacheIdx[0], cacheIdx[1], scale])
                self.cachedTiles[tile] = cacheIdx
                toRender.append( (tile, cacheIdx) )

        for (tile, cacheIdx) in toRender:
            self.renderTile(tile, cacheIdx)

        #print "updated:", len(toInsert)
