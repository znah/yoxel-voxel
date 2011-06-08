import cv, cv2
import subprocess
from numpy import *

def detect(img):
    tmp_fn = 'tmp.pgm'
    cv.SaveImage(tmp_fn, img)
    p = subprocess.Popen(['lsd.exe', tmp_fn, '-'], stdout = subprocess.PIPE)
    return fromstring(p.communicate()[0], float32, sep=' ').reshape(-1, 5)
    
def draw_segments(img, segments, color = (0, 255, 0)):
    for x1, y1, x2, y2, w in int32(segments):
        cv2.line(img, (x1, y1), (x2, y2), color, 1, cv2.CV_AA)




if __name__ == '__main__':
    import sys
    try: fn = sys.argv[1]
    except: fn = 'images/thai.jpg'

    img = cv2.imread(fn, 0)
    sg = detect(img)
    vis = cv2.cvtColor(img, cv.CV_GRAY2BGR)
    draw_segments(vis, sg)
    cv2.imshow('img', vis)
    cv2.waitKey()
