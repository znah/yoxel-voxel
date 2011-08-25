# -*- coding: utf-8 -*-
from zgl import *
import cv2
from homography import decomposeH
    
h, w = 7216, 5412
sensor_width = 49.0
focal_length = 50.0

class SaveTransform:
    def __enter__(self):
        glPushMatrix()
    def __exit__(self):
        glPopMatrix()
    

class App(ZglAppWX):
    def __init__(self):
        ZglAppWX.__init__(self, viewControl = FlyCamera())
        fragProg = genericFP('tc0')

        f = w * focal_length / sensor_width
        K = float64([[ f, 0, 0.5*w],
                     [ 0, f, 0.5*h],
                     [ 0, 0,     1]])
        Ki = linalg.inv(K)
        Himg = load('h.npy')
        H = dot(Ki, dot(Himg, K))

        print decomposeH(H)

        '''
        texs = []
        for fn in ['data/g125.jpg', 'data/g126.jpg']:
            img = cv2.imread(fn)
            img = cv2.pyrDown(cv2.pyrDown(img))
            tex = Texture2D(img)
            tex.filterLinearMipmap()
            tex.genMipmaps()
            tex.aniso(8)
        '''

    def draw_frustum(self, P, tex):
        pass
        

    def display(self):
        clearGLBuffers()
        with ctx(self.viewControl.with_vp, glstate(GL_DEPTH_TEST)):
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            drawGrid(10, 10, True)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            #self.draw_frustum()

if __name__ == "__main__":
    App().run()
