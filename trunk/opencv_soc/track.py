from numpy import *
import cv
from common import gray2bgr, anorm, anorm2

#from matplotlib import delaunay

fn = 'images/DSCN7774.MOV'

#cap = cv.CreateFileCapture(fn)
cap = cv.CreateCameraCapture(0)

frame = cv.QueryFrame(cap)
w, h = cv.GetSize(frame)
gray = cv.CreateMat(h, w, cv.CV_8U)
mask = cv.CreateMat(h, w, cv.CV_8U)

print w, h


def detect(img, mask = None):
    quality = 0.05
    min_distance = 5
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
    win_size = 3
    level = 1
    features, status, error  = cv.CalcOpticalFlowPyrLK(img0, img1, None, None, features, 
        (win_size, win_size), level, (cv.CV_TERMCRIT_EPS | cv.CV_TERMCRIT_ITER, 10, 0.03), 0, features)
    return features


life = 30

cv.CvtColor(frame, gray, cv.CV_BGR2GRAY)
tracks = [ [p] for p in detect(gray) ]


pnum = []

while True:
    prev_gray = cv.CloneMat(gray)
    cv.CvtColor(frame, gray, cv.CV_BGR2GRAY)
    
    rs = random.rand(len(tracks))
    tracks = [tr for tr, r in zip(tracks, rs) if len(tr) < life or r < 0.5]
    p0 = [tr[-1] for tr in tracks]
    p1 = track(prev_gray, gray, p0)
    p0r = track(gray, prev_gray, p1)
    good = (anorm(array(p0)-p0r) < 1.0) 

    if len(p1) > 0:
        new_tracks = []
        for tr, p, good_flag in zip(tracks, p1, good):
            if good_flag:
                tr.append(p)
                new_tracks.append(tr)
        tracks = new_tracks

    cv.Set(mask, 255)
    for x, y in int32(p1):
        cv.Circle(mask, (x, y), 5, 0, -1)
    tracks.extend( [ [p] for p in detect(gray, mask) ] )
    
    vis = cv.CloneImage(frame)#gray2bgr(gray)
    #cv.Zero(vis)
    #for x, y in int32(detect(gray)):
    #    cv.Circle(vis, (x, y), 2, (0, 255, 0), 1)
    
    '''
    x, y = array(p1).T
    circumcenters, trg_edges, tri_points, tri_neighbors = delaunay.delaunay(x, y)
    for i, j in trg_edges:
        x1, y1 = int32(p1[i])
        x2, y2 = int32(p1[j])
        cv.Line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
    '''

    for tr in tracks:
        '''
        if len(tr) < 10:
            continue
        tr = array(tr)
        d = anorm(tr[-1]-tr[-4])
        if d < 2:
            continue
        x, y = int32(tr[-1])             
        '''
        x, y = int32(tr[-1]) 
        cv.Circle(vis, (x, y), 2, (0, 255, 0), -1)

    pnum.append(len(tracks))
    #int_tracks = [ [(x, y) for x, y in int32(tr)]  for tr in tracks]
    #cv.PolyLine(vis, int_tracks, 0, (0, 255, 0))
    

    cv.ShowImage('frame', vis)
    ch = cv.WaitKey(10)
    if ch == 27:
       break
    frame = cv.QueryFrame(cap)
    if frame is None:
        cv.SetCaptureProperty(cap, cv.CV_CAP_PROP_POS_FRAMES, 0)
        frame = cv.QueryFrame(cap)

