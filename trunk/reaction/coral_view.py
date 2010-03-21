from __future__ import with_statement
from zgl import *
    
class App(ZglAppWX):
    
    dataDir = String('coral_gen/13')
    generation = Range(1, 120, 1, mode='slider')

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






    @on_trait_change('dataDir, generation')
    def loadmesh(self):
        fname = self.dataDir + '/coral_%03d.npz' % (self.generation,)
        self.mesh = load(fname)
        print fname, 'loaded'


    def __init__(self):
        ZglAppWX.__init__(self, viewControl = FlyCamera())
        self.viewControl.speed = 20

        self.loadmesh()

        code = file('data/cg/fragment_light.cg').read()
        vertProg = CGShader('vp40', code, entry = 'VertexProg')
        fragProg = CGShader('fp40', code, entry = 'FragmentProg')
        fragProg.globalAmbient = (1.0, 1.0, 1.0)
        fragProg.lightColor    = (1.0, 1.0, 1.0)
        fragProg.Ke            = (0.0, 0.0, 0.0)

        fragProg.lightPosition = (64, 64, 128)

        def draw():
            if self.animate:
              if self.time - self.lastGrowTime > 1.0 / self.growthRate:
                  self.lastGrowTime = self.time
                  if self.generation < 120:
                      self.generation += 1

            #fragProg.lightPosition = self.viewControl.eye
            fragProg.eyePosition   = self.viewControl.eye
            fragProg.Ka            = V(self.Ka[:3]) / 255.0
            fragProg.Kd            = V(self.Kd[:3]) / 255.0
            fragProg.Ks            = V(self.Ks[:3]) / 255.0
            fragProg.shininess     = self.shininess
            fragProg.lambertWrap   = V(self.lambertWrap ) / 255.0
            with ctx(self.viewControl.with_vp, vertProg, fragProg, glstate(GL_DEPTH_TEST, GL_DEPTH_CLAMP_NV)):
                drawArrays(GL_TRIANGLES, 
                  verts   = self.mesh['positions'], 
                  indices = self.mesh['faces'],
                  normals = self.mesh['normals'])
        self.draw = draw

    def key_LEFT(self):
        self.generation -=1
    def key_RIGHT(self):
        self.generation +=1
    def key_UP(self):
        self.generation = 120
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
