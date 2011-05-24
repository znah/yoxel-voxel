import numpy as np
import cv2, cv
from time import clock

import cleye

cam = cleye.CLEye(60.0)

cv2.namedWindow('camera')
cam.add_sliders('camera')

cv2.namedWindow('hist', 0)

color_samples = []

cam.set_wb(255, 255, 255)

def onmouse(event, x, y, flag, param):
    if flag & cv.CV_EVENT_FLAG_LBUTTON:
        sample = cam.frame[y, x, :3].copy()
        color_samples.append(sample)
        print sample
cv.SetMouseCallback('camera', onmouse, None)

hsv_map = np.zeros((180, 256, 3), np.uint8)
h, s = np.indices(hsv_map.shape[:2])
hsv_map[:,:,0] = h
hsv_map[:,:,1] = s
hsv_map[:,:,2] = 255
hsv_map = cv2.cvtColor(hsv_map, cv.CV_HSV2BGR)

cv2.imshow('hsv_map', hsv_map)

hist_scale = 5
def set_scale(val):
    global hist_scale
    hist_scale = val
cv.CreateTrackbar('scale', 'hist', hist_scale, 10, set_scale)

t = clock()
while True:
    frame = cam.get_frame()
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

cam.stop()

np.save('ball_colors', np.array(color_samples))
