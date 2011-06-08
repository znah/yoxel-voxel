import cv, cv2
from numpy import *
from time import clock

#cap = cv.CaptureFromCAM(1)
cap = cv.CreateCameraCapture(-1)
print cap

w, h = 640, 480

cv.SetCaptureProperty(cap, cv.CV_CAP_PROP_FRAME_WIDTH, w);
#cv.SetCaptureProperty(cap, cv.CV_CAP_PROP_FRAME_HEIGHT, h);
#cv.SetCaptureProperty(cap, cv.CV_CAP_PROP_FPS, 15.0)
#cv.SetCaptureProperty(cap, cv.CV_CAP_PROP_EXPOSURE, 10.0)

shot_idx = 0

t = 0

while True:
    frame = cv.QueryFrame(cap)
    t1 = clock()
    #print 1.0/(t1 - t)
    t = t1

    cv.ShowImage('frame', frame)
    ch = cv.WaitKey(10)
    if ch == 27:
        break
    if ch == ord(' '):
        print 'capture start ...'
        frame_n = 30*5
        a = zeros((frame_n, 480, 640, 3), uint8)
        for i in xrange(frame_n):
            #dll.CLEyeCameraGetFrame(cam, frame, 1000)
            frame = cv.QueryFrame(cap)
            a[i] = cv.GetMat(frame)
        print 'capture saving ...'
        for i in xrange(frame_n):
            cv2.imwrite('out/frame_%04d.bmp' % i, a[i])
        print 'capture done'
