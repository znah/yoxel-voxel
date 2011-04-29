from numpy import *
import cv
from common import anorm, gray2bgr

fn1 = 'images/DSCN7794.JPG'
fn2 = 'images/DSCN7796.JPG'

def scale(img, d):
    w, h = cv.GetSize(img)
    small = cv.CreateMat(h/d, w/d, cv.CV_8U)
    cv.Resize(img, small)
    return small

def detect(img):
    surf_params = (0, 200, 3, 4) # (extended, hessianThreshold, nOctaves, nOctaveLayers)
    keypoints, descriptors = cv.ExtractSURF(img, None, cv.CreateMemStorage(), surf_params)
    return keypoints, asarray(descriptors, float32)

def match(desc1, desc2):
    res = []
    for i in xrange(len(desc1)):
        dist = anorm( desc2 - desc1[i] )
        n1, n2 = dist.argsort()[:2]
        r = dist[n1] / dist[n2]
        res.append((i, n1, r))
    return res


img1 = scale(cv.LoadImage(fn1, 0), 4)
img2 = scale(cv.LoadImage(fn2, 0), 4)

k1, d1 = detect(img1)
k2, d2 = detect(img2)

matches = match(d1, d2)
matches = [(i, j) for i, j, r in matches if r < 0.7]
print len(matches)

p1 = float32( [k1[i][0] for i,j in matches] )
p2 = float32( [k2[j][0] for i,j in matches] )

F = zeros((3, 3), float32)
#status = zeros((1, len(matches)), int8)
status = cv.CreateMat(1, len(matches), cv.CV_8S)
cv.Set(status, 1)
cv.FindFundamentalMat(p1, p2, F, cv.CV_FM_LMEDS, 1.0, 0.99, status)
status = asarray(status)

print F
print sum(status)
                
vis1 = gray2bgr(img1)
vis2 = gray2bgr(img2)

for (i, j), st in zip(matches, status[0]):
    if st == 0:
        continue

    # (x, y), laplacian, size, angle, hessian
    kp0 = k1[i]
    kp1 = k2[j]

    (sx, sy), size = int32(kp0[0]), kp0[2]
    cv.Circle(vis1, (sx, sy), size/2, (0, 255, 0), 1, cv.CV_AA)
    (dx, dy), size = int32(kp1[0]), kp1[2]
    cv.Circle(vis2, (dx, dy), size/2, (0, 255, 0), 1, cv.CV_AA)
    #ex, ey = int32(dot(self.M, (sx, sy, 1)))
    #cv.Line(vis, (dx, dy), (ex, ey), (0, 0, 255), 1, cv.CV_AA)

cv.ShowImage('img1', vis1)
cv.ShowImage('img2', vis2)

cv.WaitKey(0)