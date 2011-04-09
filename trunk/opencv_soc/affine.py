from numpy import *
import cv
import sys

from common import to_list, anorm

class AffineWidget(object):
    def __init__(self, img, winname='affine'):
        self.winname = winname
        self.src = cv.GetMat(img)
        self.dst = cv.CloneMat(self.src)
        
        w, h = cv.GetSize(self.src)
        self.vis = cv.CreateMat(h, w, cv.CV_8UC3)
        
        self.M = cv.CreateMat(2, 3, cv.CV_32F)
        d = min(w, h)/6
        self.markers0 = array([[w/2, h/2], [w/2+d, h/2], [w/2, h/2-d]], float32)
        self.markers = self.markers0.copy()

        self.mouse_state = self.onmouse_wait
        self.transform()

    def onmouse(self, event, x, y, flags, param):
        self.mouse_state(event, x, y, flags)

    def onmouse_wait(self, event, x, y, flags):
        if event == cv.CV_EVENT_LBUTTONDOWN:
            d = anorm(self.markers - (x, y))
            if d.min() > 8:
                return
            self.drag_mark_idx = d.argmin()
            self.mouse_state = self.onmouse_drag
        if event == cv.CV_EVENT_RBUTTONDOWN:
            self.markers[:] = self.markers0
            self.update()

    def onmouse_drag(self, event, x, y, flags):
        if flags & cv.CV_EVENT_FLAG_LBUTTON == 0:
            self.mouse_state = self.onmouse_wait

        if self.drag_mark_idx == 0:
            self.markers += (x, y) - self.markers[0]
        else:
            if flags & cv.CV_EVENT_FLAG_SHIFTKEY:
                self.markers[self.drag_mark_idx] = x, y
            else:
                center = self.markers[0]
                x0, y0 = self.markers[self.drag_mark_idx] - center
                x1, y1 = (x, y) - center
                try:
                    a, b = linalg.solve([[x0, -y0], [y0, x0]], (x1, y1))
                    A = array([[a, -b], [b, a]])
                    for i in [1, 2]:
                        self.markers[i] = dot(A, self.markers[i] - center) + center
                except linalg.LinAlgError:
                    self.markers[self.drag_mark_idx] = x, y
        self.update()

    def update(self):
        self.transform()
        self.show()
    
    def transform(self):
        cv.GetAffineTransform(to_list(self.markers0), to_list(self.markers), self.M)
        cv.WarpAffine(self.src, self.dst, self.M)

    def draw_overlay(self, vis):
        markers = to_list(int32(self.markers))
        col = (0, 255, 255)
        for p in markers:
            cv.Circle(vis, p, 5, col, 2, cv.CV_AA)
        cv.Line(vis, markers[0], markers[1], col, 2, cv.CV_AA)
        cv.Line(vis, markers[0], markers[2], col, 2, cv.CV_AA)

    def show(self):
        if self.dst.channels == 1:
            cv.CvtColor(self.dst, self.vis, cv.CV_GRAY2BGR)
        else:
            cv.Copy(self.dst, self.vis)
        
        self.draw_overlay(self.vis)
                
        cv.ShowImage(self.winname, self.vis)
        cv.SetMouseCallback(self.winname, self.onmouse)

if __name__ == '__main__':
    print 'INSTRUCTION: drag yellow markers to transform the image, use SHIFT to skew it'

    fn = 'sn.jpg'
    if len(sys.argv) > 1:
        fn = sys.argv[1]

    img = cv.LoadImage(fn)

    cv.NamedWindow('affine', 0)
    aw = AffineWidget(img, winname='affine')
    aw.show()
    cv.WaitKey(0)
