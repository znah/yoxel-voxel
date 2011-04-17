from numpy import *
import cv
from common import gray2bgr

fn = 'images/DSCN7774.MOV'

cap = cv.CreateFileCapture(fn)

frame = cv.QueryFrame(cap)
w, h = cv.GetSize(frame)
gray = cv.CreateMat(h, w, cv.CV_8U)
mask = cv.CreateMat(h, w, cv.CV_8U)


def detect(img, mask = None):
    quality = 0.07
    min_distance = 3
    MAX_COUNT = 1000
    block = 5
    #win_size = 5
    
    features = cv.GoodFeaturesToTrack (
       img,
       None, None, #eig, temp,
       MAX_COUNT,
       quality, min_distance, mask, block, 0)
    return features

def track(img0, img1, features):
    win_size = 5
    level = 1
    features, status, error  = cv.CalcOpticalFlowPyrLK(img0, img1, None, None, features, 
        (win_size, win_size), level, (cv.CV_TERMCRIT_EPS | cv.CV_TERMCRIT_ITER, 10, 0.03), 0, features)
    return features


life = 20

cv.CvtColor(frame, gray, cv.CV_BGR2GRAY)
tracks = [ [p] for p in detect(gray) ]

fnum = [len(tracks)]

while True:
    prev_gray = cv.CloneMat(gray)
    cv.CvtColor(frame, gray, cv.CV_BGR2GRAY)
    
    rs = random.rand(len(tracks))
    tracks = [tr for tr, r in zip(tracks, rs) if len(tr) < life]
    p0 = [tr[-1] for tr in tracks]
    p1 = track(prev_gray, gray, p0)
    for tr, p in zip(tracks, p1):
        tr.append(p)

    cv.Set(mask, 255)
    for x, y in p1:
        cv.Circle(mask, (x, y), 3, 0, -1)
    cv.ShowImage('mask', mask)
    
    tracks.extend( [ [p] for p in detect(gray, mask) ] )
    fnum.append(len(tracks))

    vis = gray2bgr(gray)
    cv.Zero(vis)
    for x, y in p1:
        cv.Circle(vis, (x, y), 2, 255, -1)
    #cv.PolyLine(vis, tracks, 0, (0, 255, 0))
    

    cv.ShowImage('frame', vis)
    ch = cv.WaitKey(10)
    if ch == 27:
       break
    frame = cv.QueryFrame(cap)
    if frame is None:
        cv.SetCaptureProperty(cap, cv.CV_CAP_PROP_POS_FRAMES, 0)
        frame = cv.QueryFrame(cap)

