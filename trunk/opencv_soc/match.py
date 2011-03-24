from numpy import *
import cv

from affine import AffineWidget

from common import gray2bgr, anorm

class DetectApp(AffineWidget):
    def __init__(self, img):
        self.src_keypoints, self.src_descriptors = self.detect(img)
        self.dst_keypoints = None
        
        AffineWidget.__init__(self, img, 'dst')

        vis = gray2bgr(self.src)
        self.draw_keypoints(vis, self.src_keypoints)
        
        cv.ShowImage('src', vis)
        self.show()

    def transform(self):
        AffineWidget.transform(self)

        if self.mouse_state != self.onmouse_wait:
            self.dst_keypoints = None
            return
        
        dst_keypoints, dst_descriptors = self.detect(self.dst)
        matches = self.match(self.src_descriptors, dst_descriptors)
        print len(matches), len(self.src_keypoints)
        self.dst_keypoints = []
        for i, j in matches:
            self.dst_keypoints.append(dst_keypoints[j])


    def draw_overlay(self, vis):
        AffineWidget.draw_overlay(self, vis)
        if self.dst_keypoints is not None:
            self.draw_keypoints(vis, self.dst_keypoints)

    def detect(self, img):
        self.surf_params = (0, 2000, 3, 4) # (extended, hessianThreshold, nOctaves, nOctaveLayers)
        keypoints, descriptors = cv.ExtractSURF(img, None, cv.CreateMemStorage(), self.surf_params)
        return keypoints, asarray(descriptors, float32)
     
    def match(self, desc1, desc2):
        res = []
        for i in xrange(len(desc1)):
            dist = anorm( desc2 - desc1[i] )
            n1, n2 = dist.argsort()[:2]
            if dist[n1] / dist[n2] < 0.5:
                res.append((i, n1))
        return res

    def draw_keypoints(self, vis, keypoints):
        for (x, y), laplacian, size, angle, hessian in keypoints:
            cv.Circle(vis, (int(x), int(y)), size/2, (0, 255, 0))


        

            
        

if __name__ == '__main__':
    import sys

    fn = 'sn.jpg'
    if len(sys.argv) > 1:
        fn = sys.argv[1]
    
    img = cv.LoadImage(fn, 0)

    aw = DetectApp(img)
    cv.WaitKey(0)