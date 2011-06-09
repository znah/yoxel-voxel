import numpy as np
import cv2, cv
from time import clock
import sys

import video

hsv_map = np.zeros((180, 256, 3), np.uint8)
h, s = np.indices(hsv_map.shape[:2])
hsv_map[:,:,0] = h
hsv_map[:,:,1] = s
hsv_map[:,:,2] = 255
hsv_map = cv2.cvtColor(hsv_map, cv.CV_HSV2BGR)
cv2.imshow('hsv_map', hsv_map)

cv2.namedWindow('hist', 0)
hist_scale = 5
def set_scale(val):
    global hist_scale
    hist_scale = val
cv.CreateTrackbar('scale', 'hist', hist_scale, 10, set_scale)

try: fn = sys.argv[1]
except: fn = 'synth:bg=../cpp/lena.jpg:noise=0.1'
cam = video.create_capture(fn)

t = clock()
while True:
    flag, frame = cam.read()
    cv2.imshow('camera', frame)
    
    small = cv2.pyrDown(frame)

    hsv = cv2.cvtColor(small, cv.CV_BGR2HSV)
    h = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )


    h = np.clip(h*0.005*hist_scale, 0, 1)
    vis = hsv_map*h[:,:,np.newaxis] / 255.0
    cv2.imshow('hist', vis)
    

    t1 = clock()
    #print (t1-t)*1000
    t = t1

    ch = cv2.waitKey(1)
    if ch == 27:
        break
