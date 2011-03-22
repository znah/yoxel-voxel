import sys
import cv


class AffineWidget:
    def __init__(self, img, winname='affine'):
        self.winname = winname
        self.src = cv.GetMat(img)
        self.dst = cv.CloneMat(self.src)
        
        w, h = cv.GetSize(self.src)
        self.vis = cv.CreateMat(h, w, cv.CV_8UC3)
        
        self.M = M = cv.CreateMat(2, 3, cv.CV_32F)
        cv.SetZero(M)
        M[0, 0] = 1
        M[1, 1] = 1
    
    def transform(self):
        cv.WarpAffine(self.src, self.dst, self.M)

    def show(self):
        self.transform()
        cv.Copy(self.dst, self.vis)	
                
        cv.NamedWindow(self.winname, 0)
        cv.ShowImage(self.winname, self.vis)

    def onmouse(self, event, x, y, flags, param):
        pass
     



fn = 'sn.jpg'
if len(sys.argv) > 1:
    fn = sys.argv[1]

img = cv.LoadImage(fn)

aw = AffineWidget(img)
aw.show()

while cv.WaitKey(0) == chr(27):
    pass

