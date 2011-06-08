import numpy as np
import cv2, cv
import lsd

from common import anorm, homotrans, rect2rect_mtx

from numpy.linalg import inv

pixel_extent = 0.1 # meters


def match(desc1, desc2, r_threshold = 0.6):
    res = []
    for i in xrange(len(desc1)):
        dist = anorm( desc2 - desc1[i] )
        n1, n2 = dist.argsort()[:2]
        r = dist[n1] / dist[n2]
        if r < r_threshold:
            res.append((i, n1))
    return np.array(res)


def match_imgs(img1, img2, thrs = 500, ratio_thrs = 0.5):
    gray1 = cv2.cvtColor(img1, cv.CV_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv.CV_BGR2GRAY)

    surf = cv2.SURF(thrs, 2, 3, _upright=True)
    p1, d1 = surf.detect(gray1, None, False)
    p2, d2 = surf.detect(gray2, None, False)
    d1.shape = (-1, surf.descriptorSize())
    d2.shape = (-1, surf.descriptorSize())
    m = match(d1, d2, ratio_thrs)
    print len(m)
    
    vis = img1.copy()
    for i, j in m:
        v1 = tuple(np.int32( p1[i].pt )) 
        v2 = tuple(np.int32( p2[j].pt ))
        cv2.circle(vis, v1, 2, (0, 255, 0), -1)
        cv2.line(vis, v1, v2, (0, 255, 0))
    cv2.imshow('match', vis)


class Frame:
    def __init__(self, fn):
        self.fn = fn

        self.lods = lods = [cv2.imread(fn)]
        for i in xrange(4):
            lods.append( cv2.pyrDown(lods[-1]) )
        print fn, 'loaded'

        surf = cv2.SURF(3000, 2, 2)

        self.feature_lod = 4
        feature_img = cv2.cvtColor(lods[ self.feature_lod ], cv.CV_BGR2GRAY)
        self.keypoints_lod, self.desc = surf.detect(feature_img, None, False)
        self.desc.shape = (-1, surf.descriptorSize())

        self.keypoints = np.float32( [kp.pt for kp in self.keypoints_lod] ) * 2**self.feature_lod

        h, w = lods[0].shape[:2]
        s = 2.0 / w
        self.img2norm = np.array([[ s, 0, -1.0],
                                  [ 0, s, -1.0*h/w],
                                  [ 0, 0,  1.0]])

        print len(self.keypoints), 'features detected'


def find_homography(frame1, frame2):
    m1, m2 = match(frame1.desc, frame2.desc).T
    p1 = np.float32(frame1.keypoints[m1].copy())
    p2 = np.float32(frame2.keypoints[m2].copy())
    
    H, mask = cv2.findHomography(p1, p2, cv.CV_RANSAC, 20.0 / pixel_extent)
    print '%d / %d matched' % (mask.sum(), mask.size)
    mask = np.ravel(mask != 0)
    return H, p1[mask], p2[mask]

class App:
    def __init__(self):
        names = ['data/g125.jpg', 'data/g126.jpg']
        #names = ['112.jpg', '113.jpg']
        frames = [Frame(fn) for fn in names]
        H12, p1, p2 = find_homography(frames[0], frames[1])

        preview_size = (512, 512)
        extent = (-1000, -1000, 8000, 8000)
        wld2prev = rect2rect_mtx(extent, preview_size)
        
        vis2 = cv2.warpPerspective(frames[1].lods[0], wld2prev, preview_size)
        vis1 = cv2.warpPerspective(frames[0].lods[0], np.dot(wld2prev, H12), preview_size)

        preview = vis1/2 + vis2/2
        cv2.imshow('preview', preview)

        def onmouse(event, x, y, flags, param):
            if flags & cv.CV_EVENT_FLAG_LBUTTON == 0:
                return
            cx, cy = np.dot((x, y, 1), inv(wld2prev).T)[:2]
            w, h = 800, 600
            wld2view = rect2rect_mtx([cx-w/2, cy-h/2, cx+w/2, cy+h/2], (w, h))

            vis2 = cv2.warpPerspective(frames[1].lods[0], wld2view, (w, h))
            vis1 = cv2.warpPerspective(frames[0].lods[0], np.dot(wld2view, H12), (w, h))
            view = vis1/2 + vis2/2
            cv2.imshow('view', view)

            match_imgs(vis1, vis2)    

        cv.SetMouseCallback('preview', onmouse, None)

    
        
    def run(self):
        cv2.waitKey()
        

if __name__ == '__main__':
    App().run()
