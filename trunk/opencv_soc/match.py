from numpy import *
import cv

from affine import AffineWidget

class DetectApp(AffineWidget):
    def __init__(self, img):
        AffineWidget.__init__(self, img, 'dst')

        cv.ShowImage('src', self.src)
        self.show()

    def transform(self):
        AffineWidget.transform(self)




        

if __name__ == '__main__':
    import sys

    fn = 'sn.jpg'
    if len(sys.argv) > 1:
        fn = sys.argv[1]
    
    img = cv.LoadImage(fn, 0)

    aw = DetectApp(img)
    cv.WaitKey(0)