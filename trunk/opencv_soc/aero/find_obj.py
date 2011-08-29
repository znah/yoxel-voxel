# -*- coding: utf-8 -*-

import numpy as np
import cv2
from common import anorm

import pose

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing

flann_params = dict(algorithm = FLANN_INDEX_KDTREE,
                    trees = 4)

def match(desc1, desc2, r_threshold = 0.75):
    res = []
    for i in xrange(len(desc1)):
        dist = anorm( desc2 - desc1[i] )
        n1, n2 = dist.argsort()[:2]
        r = dist[n1] / dist[n2]
        if r < r_threshold:
            res.append((i, n1))
    return np.array(res)


def match_flann(desc1, desc2, r_threshold = 0.6):
    flann = cv2.flann_Index(desc2, flann_params)
    idx2, dist = flann.knnSearch(desc1, 2, params = {}) # bug: need to provide empty dict
    mask = dist[:,0] / dist[:,1] < r_threshold
    idx1 = np.arange(len(desc1))
    pairs = np.int32( zip(idx1, idx2[:,0]) )
    return pairs[mask]

def draw_features(img, points, sizes):
    img = cv2.cvtColor( img, cv2.cv.CV_GRAY2BGR )
    for (x, y), r in zip( np.int32( points.reshape(-1, 2) ), sizes ):
        #r = 3
        cv2.circle(img, (x, y), int(r/2), (0, 255, 0), 2, cv2.CV_AA)
    return img

def load_img(fn):
    img = cv2.imread(fn, 0)
    print fn
    for i in xrange(2):
        img = cv2.pyrDown(img)
    return img


def dot3(A, B, C):
    return np.dot(A, np.dot(B, C))

if __name__ == '__main__':
    import sys
    try: fn1, fn2 = sys.argv[1:3]
    except:
        fn1 = 'data/karmanov/house1.jpg'
        fn2 = 'data/karmanov/house2.jpg'

    img1 = load_img(fn1)
    img2 = load_img(fn2)

    surf = cv2.SURF(200, 2, 2, 0, False)
    kp1, desc1 = surf.detect(img1, None, False)
    kp2, desc2 = surf.detect(img2, None, False)
    desc1.shape = (-1, surf.descriptorSize())
    desc2.shape = (-1, surf.descriptorSize())
    print 'img1 - %d features, img2 - %d features' % (len(kp1), len(kp2))

    m = match_flann(desc1, desc2)
    matched_p1 = np.array([kp1[i].pt for i, j in m])
    matched_p2 = np.array([kp2[j].pt for i, j in m])
    r1 = np.array([kp1[i].size for i, j in m])
    r2 = np.array([kp2[j].size for i, j in m])
    H, Hstatus = cv2.findHomography(matched_p1, matched_p2, cv2.RANSAC, 5.0)
    F, Fstatus = cv2.findFundamentalMat(matched_p1, matched_p2, cv2.FM_RANSAC, 1.0)
    status = Fstatus
    print '%d / %d  inliers/matched' % (np.sum(status), len(status))
    status = status.ravel() > 0
    in_p1 = matched_p1[status]
    in_p2 = matched_p2[status]
    
    vis1 = draw_features(img1, in_p1, r1[status])
    vis2 = draw_features(img2, in_p2, r2[status])
    #cv2.imwrite('vis1.bmp', vis1)
    #cv2.imwrite('vis2.bmp', vis2)

    cv2.imshow('img1', vis1)
    cv2.imshow('img2', vis2)

    #cv2.imwrite('small1.jpg', img1)
    #cv2.imwrite('small2.jpg', img2)


    K = pose.makeK((4000, 3000), 6.0, 8.08)
    Ki = np.linalg.inv(K)
    pos1h, pos2h = pose.decomposeH(dot3(K, H, Ki))
    pos1e, pos2e =  pose.decomposeE(dot3(K, F, Ki))


    def show_epiline(event, x, y, flags, param):
        a, b, c = np.dot([x, y, 1.0], F.T)
        vis = vis2.copy()
        y1 = -c/b
        y2 = -(4000*a + c)/b
        cv2.line(vis, (0, int(y1)), (4000, int(y2)), (0, 255, 255))
        cv2.imshow('img2', vis)

    cv2.setMouseCallback('img1', show_epiline)

    cv2.waitKey()

