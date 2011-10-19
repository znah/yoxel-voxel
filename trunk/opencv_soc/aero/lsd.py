import numpy as np
import cv2
import subprocess

def detect(img):
    tmp_fn = 'tmp.pgm'
    cv2.imwrite(tmp_fn, img)
    p = subprocess.Popen(['lsd.exe', tmp_fn, '-'], stdout = subprocess.PIPE)
    return np.fromstring(p.communicate()[0], np.float32, sep=' ').reshape(-1, 5)

#def draw_segments(img, segments): 
    

if __name__ == '__main__':
    import sys
    try: fn = sys.argv[1]
    except: fn = 'data/karmanov/DSCF4726.JPG'

    img = cv2.imread(fn, 0)
    sg = detect(img)

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2, w in np.int32(sg):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.CV_AA)
    
    cv2.imshow('img', vis)
    cv2.waitKey(0)
