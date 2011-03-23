from numpy import *
import cv
import sys

def to_list(a):
    return [tuple(p) for p in a]

class AffineWidget:
    def __init__(self, img, winname='affine'):
        self.winname = winname
        self.src = cv.GetMat(img)
        self.dst = cv.CloneMat(self.src)
        
        w, h = cv.GetSize(self.src)
        self.vis = cv.CreateMat(h, w, cv.CV_8UC3)
        
        self.M = cv.CreateMat(2, 3, cv.CV_32F)
        d = min(w, h)/4
        self.markers0 = array([[w/2, h/2], [w/2+d, h/2], [w/2, h/2+d]], float32)
        self.markers = self.markers0.copy()
        self.transform()
    
    def transform(self):
        cv.GetAffineTransform(to_list(self.markers0), to_list(self.markers), self.M)
        cv.WarpAffine(self.src, self.dst, self.M)

    def show(self):
        cv.Copy(self.dst, self.vis)	
                
        cv.NamedWindow(self.winname, 0)
        cv.ShowImage(self.winname, self.vis)
        cv.SetMouseCallback(self.winname, self.onmouse)

    def onmouse(self, event, x, y, flags, param):
        self.markers[2] = x, y
        self.transform()
        self.show()
     



fn = 'sn.jpg'
if len(sys.argv) > 1:
    fn = sys.argv[1]

img = cv.LoadImage(fn)

aw = AffineWidget(img)
aw.show()

while cv.WaitKey(0) == chr(27):
    pass

