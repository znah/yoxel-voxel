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
    
def makeP(K, R, t):
    P = eye(4)
    P[:3, :3] = R
    P[:3,  3] = t
    P[:3, :4] = dot(K, P[:3, :4])
    m = eye(4)
    m[2:, 2:] = 1 - m[2:, 2:]
    return dot(m, P)
    

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

        fx = 2.0 * focal_length / sensor_width
        fy = fx / w * h
        Kdraw = diag([fx, fy, 1.0])

        #sols = []
        #for R, t, n in decomposeH(H):
        #    P1, P2 = place_cameras(R, t, n, Kdraw)

        #print decomposeH(H)

        verts, trg_idxs, quad_idxs = create_box()
        v = ones((len(verts), 4))
        v[:,:3] = verts
        P = makeP(Kdraw, eye(3), [0, 0, 0])
        v = dot( v, linalg.inv(P).T )
        #v /= v[:,3, newaxis]
        def draw_box():
            #glColor(0.5, 0.5, 0.5)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            drawArrays(GL_QUADS, verts = v, indices = quad_idxs)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self.draw_box = draw_box
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

    def display(self):
        clearGLBuffers()
        with ctx(self.viewControl.with_vp, glstate(GL_DEPTH_TEST)):
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            drawGrid(10, 10, True)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            self.draw_box()
            #self.draw_frustum()

if __name__ == "__main__":
    App().run()
