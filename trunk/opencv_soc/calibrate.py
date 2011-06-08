from numpy import *
import cv
from glob import glob
import os

from common import gray2bgr
 
pattern_size = (8, 6)
chess_flags = cv.CV_CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_FAST_CHECK | cv.CV_CALIB_CB_FILTER_QUADS
term_crit = (cv.CV_TERMCRIT_EPS | cv.CV_TERMCRIT_ITER, 30, 0.1)

pattern_points = [(j, i, 0) for i, j in ndindex(pattern_size[1], pattern_size[0])]

obj_points = []
img_points = []
point_counts = []
img_size = None

names = glob('out/shot_*.bmp')
for fn in names:
    img = cv.LoadImage(fn, 0)
    img_size = cv.GetSize(img)
    found, corners = cv.FindChessboardCorners(img, pattern_size, chess_flags)
    vis = gray2bgr(img)
    if found:
        corners = cv.FindCornerSubPix(img, corners, (8, 8), (-1, -1), term_crit)
        obj_points.extend( pattern_points )
        img_points.extend( corners )
        point_counts.append( len(pattern_points) )

        print fn
        cv.DrawChessboardCorners(vis, pattern_size, corners, found)
        debug_fn = 'out/chess_%s.bmp' % os.path.splitext(fn)[0][-4:]
        cv.SaveImage(debug_fn, vis)


K = zeros((3, 3), float32)
dist_coef = zeros((1, 5), float32)
M = len(point_counts)
rvecs = zeros((M, 3), float32)
tvecs = zeros((M, 3), float32)
cv.CalibrateCamera2(float32(obj_points), float32(img_points), int32(point_counts).reshape(1, -1), 
                     img_size, K, dist_coef,
                     rvecs, tvecs, 0 )


print K, dist_coef

'''    

cap = cv.CreateCameraCapture(0)




while True:
    frame = cv.QueryFrame(cap)
    found, corners = cv.FindChessboardCorners(frame, pattern_size, chess_flags)
    
    vis = cv.CloneImage(frame)
    if found:
        cv.DrawChessboardCorners(vis, pattern_size, corners, found)
    else:
        for x, y in int32(corners):
            cv.Circle(vis, (x, y), 2, (0, 255, 0), -1)

    
    cv.ShowImage('frame', vis)

    ch = cv.WaitKey(10)
    if ch == 27:
        break

'''