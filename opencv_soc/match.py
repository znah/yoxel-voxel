from numpy import *
import cv

from affine import AffineWidget

from common import gray2bgr, anorm

class DetectApp(AffineWidget):
    def __init__(self, img):
        AffineWidget.__init__(self, img, 'dst')
        
        self.src_keypoints, self.src_descriptors = self.detect(self.src)
        self.matches = []

        self.update()

    def transform(self):
        AffineWidget.transform(self)

        if not hasattr(self, 'src_keypoints'):
            return
        if self.mouse_state != self.onmouse_wait:
            self.matches = []
            return
        
        self.dst_keypoints, self.dst_descriptors = self.detect(self.dst)
        self.matches = self.match(self.src_descriptors, self.dst_descriptors)


    def draw_overlay(self, vis):
        AffineWidget.draw_overlay(self, vis)
        if not hasattr(self, 'src_keypoints') or len(self.matches) == 0:
            return


        matches = [(i, j) for i, j, r in self.matches if r < 0.6]
        print '%d / %d matched' % (len(matches), len(self.src_keypoints))
        
        src_vis = gray2bgr(self.src)
        for i, j in matches:
            
            # (x, y), laplacian, size, angle, hessian
            kp0 = self.src_keypoints[i]
            kp1 = self.dst_keypoints[j]

            (sx, sy), size = int32(kp0[0]), kp0[2]
            cv.Circle(src_vis, (sx, sy), size/2, (0, 255, 0), 1, cv.CV_AA)
            (dx, dy), size = int32(kp1[0]), kp1[2]
            cv.Circle(vis, (dx, dy), size/2, (0, 255, 0), 1, cv.CV_AA)
            ex, ey = int32(dot(self.M, (sx, sy, 1)))
            cv.Line(vis, (dx, dy), (ex, ey), (0, 0, 255), 1, cv.CV_AA)

        
        cv.ShowImage('src', src_vis)

    def detect(self, img):
        self.surf_params = (0, 2000, 3, 4) # (extended, hessianThreshold, nOctaves, nOctaveLayers)
        keypoints, descriptors = cv.ExtractSURF(img, None, cv.CreateMemStorage(), self.surf_params)
        return keypoints, asarray(descriptors, float32)
     
    def match(self, desc1, desc2):
        res = []
        for i in xrange(len(desc1)):
            dist = anorm( desc2 - desc1[i] )
            n1, n2 = dist.argsort()[:2]
            r = dist[n1] / dist[n2]
            res.append((i, n1, r))
        return res

if __name__ == '__main__':
    import sys

    fn = 'images/sn.jpg'
    if len(sys.argv) > 1:
        fn = sys.argv[1]
    
    img = cv.LoadImage(fn, 0)

    aw = DetectApp(img)
    cv.WaitKey(0)