import cv
import subprocess
from numpy import *

from common import gray2bgr

def detect(img):
    tmp_fn = 'tmp.pgm'
    cv.SaveImage(tmp_fn, img)
    p = subprocess.Popen(['lsd.exe', tmp_fn, '-'], stdout = subprocess.PIPE)
    return fromstring(p.communicate()[0], float32, sep=' ').reshape(-1, 5)
    
    

if __name__ == '__main__':
    import sys
    try: fn = sys.argv[1]
    except: fn = 'thai.jpg'

    img = cv.LoadImage(fn, 0)
    sg = detect(img)

    vis = gray2bgr(img)
    for x1, y1, x2, y2, w in sg:
        cv.Line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1, cv.CV_AA)
    
    #cv.NamedWindow('img', 0)
    cv.ShowImage('img', vis)
    cv.WaitKey(0)
