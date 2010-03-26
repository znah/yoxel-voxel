from __future__ import with_statement
from zgl import *
import os
    
class App(ZglAppWX):
    
    dataDir = String('coral_gen/256_1.5')
    firstGeneration = Int(0)
    lastGeneration  = Int(120)
    generation = Range('firstGeneration', 'lastGeneration', 0, mode='slider')

    Ka            = Color((0, 0, 0))
    Kd            = Color((250, 245, 255))
    Ks            = Color((54, 54, 54))
    shininess     = Range(1.0, 200.0, 100.0, mode='slider') 
    lambertWrap   = Color((255, 236, 204))

    animate       = Bool(False)
    growthRate    = Float(20)

    traits_view = View( Item(name = 'generation'),
                        Item(name = 'dataDir'),
                        Item(name='Ka', style='text'),
                        Item(name='Kd', style='text'),
                        Item(name='Ks', style='text'),
                        Item(name='shininess'),
                        Item(name='lambertWrap', style='text'),
                        resizable = True)


    @on_trait_change('dataDir')
    def scan_dir(self):
        fs = [s for s in os.listdir(self.dataDir) if s[-4:] == '.npz']
        self.genFiles = sort(fs)
        self.lastGeneration = len(self.genFiles)-1
        self.generation = min(self.lastGeneration, self.generation)
    

    @on_trait_change('generation')
    def loadmesh(self):
        fname = self.dataDir + '/' + self.genFiles[self.generation]
        self.mesh = load(fname)
        print fname, 'loaded'
        print len(self.mesh['positions'])
        self.verts   = self.mesh['positions'].copy()
        self.indices = self.mesh['faces'].copy()
        self.normals = self.mesh['normals'].copy()



    def __init__(self):
        ZglAppWX.__init__(self, viewControl = FlyCamera())
        self.viewControl.speed = 0.2
        self.viewControl.zNear = 0.01
        self.viewControl.zFar  = 10.0

        self.scan_dir()
        self.loadmesh()

        code = file('data/cg/fragment_light.cg').read()
        vertProg = CGShader('vp40', code, entry = 'VertexProg')
        fragProg = CGShader('fp40', code, entry = 'FragmentProg')
        fragProg.globalAmbient = (1.0, 1.0, 1.0)
        fragProg.lightColor    = (1.0, 1.0, 1.0)
        fragProg.Ke            = (0.0, 0.0, 0.0)

        fragProg.lightPosition = (0, 0, 1.0)

        def draw():
            if self.animate:
              if self.time - self.lastGrowTime > 1.0 / self.growthRate:
                  self.lastGrowTime = self.time
                  if self.generation < self.lastGeneration:
                      self.generation += 1

            #fragProg.lightPosition = self.viewControl.eye
            fragProg.eyePosition   = self.viewControl.eye
            fragProg.Ka            = V(self.Ka[:3]) / 255.0
            fragProg.Kd            = V(self.Kd[:3]) / 255.0
            fragProg.Ks            = V(self.Ks[:3]) / 255.0
            fragProg.shininess     = self.shininess
            fragProg.lambertWrap   = V(self.lambertWrap ) / 255.0
            with ctx(self.viewControl.with_vp, vertProg, fragProg, glstate(GL_DEPTH_TEST)):
                drawArrays(GL_TRIANGLES, 
                  verts   = self.verts, 
                  indices = self.indices,
                  normals = self.normals)
        self.draw = draw

    def key_LEFT(self):
        self.generation -=1
    def key_RIGHT(self):
        self.generation +=1
    def key_UP(self):
        self.generation = self.lastGeneration
    def key_DOWN(self):
        self.generation = 1
    def key_SPACE(self):
        self.animate = not self.animate
        self.lastGrowTime = self.time

    def display(self):
        clearGLBuffers()
        self.draw()

if __name__ == "__main__":
    App().run()
