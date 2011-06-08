from numpy import *
import cv
import sys

try: fn = sys.argv[1]
except: fn = 'right05.jpg'

src = cv.LoadImage(fn, 1)
dst = cv.CloneImage(src)

w, h = cv.GetSize(src)
K = array([[w, 0, 0.5*w], 
           [0, w, 0.5*h], 
           [0, 0,     1]])

def update(dummy):
    cv.ShowImage('lens', dst)
    k1 = (cv.GetTrackbarPos('k1', 'lens') - 50) / 25.0
    k2 = (cv.GetTrackbarPos('k2', 'lens') - 50) / 25.0
    dist_coef = array([[k1, k2, 0, 0]])
    cv.Undistort2(src, dst, K, dist_coef)
    print 'k1 = %f,  k2 = %f' % (k1, k2)

    cv.ShowImage('lens', dst)


cv.NamedWindow('lens')
cv.CreateTrackbar('k1', 'lens', 50, 100, update)
cv.CreateTrackbar('k2', 'lens', 50, 100, update)

update(0)


cv.WaitKey(0)


