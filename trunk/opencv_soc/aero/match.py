import numpy as np
import cv2, cv
import lsd

from common import anorm, homotrans, rect2rect_mtx

from scipy.spatial import KDTree

from numpy.linalg import inv

pixel_extent = 0.1 # meters

import stereo


def match(desc1, desc2, r_threshold = 0.6):
    res = []
    for i in xrange(len(desc1)):
        dist = anorm( desc2 - desc1[i] )
        n1, n2 = dist.argsort()[:2]
        r = dist[n1] / dist[n2]
        if r < r_threshold:
            res.append((i, n1))
    return np.array(res)


def local_match(p1, p2, desc1, desc2, max_dist, min_neigh = 5, r_threshold = 0.5):
    kd1 = KDTree(p1)
    kd2 = KDTree(p2)
    
    res = []
    query_res = kd1.query_ball_tree(kd2, max_dist)
    for i, d1 in enumerate(desc1):
        neigh = query_res[i]
        if len(neigh) < min_neigh:
            continue
        dist = anorm(desc2[neigh] - d1)
        n1, n2 = dist.argsort()[:2]
        r = dist[n1] / dist[n2]
        if r < r_threshold:
            res.append((i, neigh[n1]))
    return np.array(res) #match(desc1, desc2, r_threshold)


ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    verts = verts[verts[:,2] > verts[:,2].min()]
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')


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
        fnames = ['data/g125.jpg', 'data/g126.jpg']
        #fnames = ['vish/IM (25).JPG', 'vish/IM (26).JPG']
        #fnames = ['data/racurs/48.tif', 'data/racurs/49.tif']
        self.frames = frames = [Frame(fn) for fn in fnames]
        H12, p1, p2 = find_homography(frames[0], frames[1])

        preview_size = (800, 800)
        extent = (-1000, -1000, 8000, 8000)
        wld2prev = rect2rect_mtx(extent, preview_size)
        
        vis2 = cv2.warpPerspective(frames[1].lods[0], wld2prev, preview_size)
        vis1 = cv2.warpPerspective(frames[0].lods[0], np.dot(wld2prev, H12), preview_size)

        preview = vis1/2 + vis2/2
        cv2.imshow('preview', preview)

        self.shot_idx = 0

        def onmouse(event, x, y, flags, param):
            if event != cv.CV_EVENT_LBUTTONDOWN:
                return
            cx, cy = np.dot((x, y, 1), inv(wld2prev).T)[:2]
            w, h = 800, 800
            wld2view = rect2rect_mtx([cx-w/2, cy-h/2, cx+w/2, cy+h/2], (w, h))

            self.cur_preview = preview.copy()
            view2preview = np.dot(wld2prev, inv(wld2view))
            x0, y0 = np.int32( np.dot((0, 0, 1), view2preview.T)[:2] )
            x1, y1 = np.int32( np.dot((w, h, 1), view2preview.T)[:2] )
            cv2.rectangle(self.cur_preview, (x0, y0), (x1, y1), (255, 255, 255))
            cv2.imshow('preview', self.cur_preview)

            H2 = wld2view
            H1 = np.dot(wld2view, H12)
            vis2 = cv2.warpPerspective(frames[1].lods[0], H2, (w, h))
            vis1 = cv2.warpPerspective(frames[0].lods[0], H1, (w, h))

            self.match_stereo(vis1, vis2, H1, H2)    

        cv.SetMouseCallback('preview', onmouse, None)


    def match_stereo(self, img1, img2, img1_view, img2_view, thrs = 500, ratio_thrs = 0.5):
        gray1 = cv2.cvtColor(img1, cv.CV_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv.CV_BGR2GRAY)

        surf = cv2.SURF(thrs, 1, 4)
        kp1, d1 = surf.detect(gray1, None, False)
        kp2, d2 = surf.detect(gray2, None, False)
        d1.shape = (-1, surf.descriptorSize())
        d2.shape = (-1, surf.descriptorSize())
        print len(kp1), len(kp2)
        #m = match(d1, d2, ratio_thrs)
        p1 = [p.pt for p in kp1]
        p2 = [p.pt for p in kp2]
        m = local_match(p1, p2, d1, d2, max_dist = 5.0 / pixel_extent, min_neigh = 5, r_threshold = ratio_thrs)
        
        pairs = np.float32( [(kp1[i].pt, kp2[j].pt) for i, j in m] )
        mp1, mp2 = pairs[:,0].copy(), pairs[:,1].copy()

        '''
        for (x1, y1), (x2, y2) in np.int32(zip(mp1, mp1)):
            cv.circle(img1, (x1, y1), 2, (0, 255, 0))
            cv.circle(img2, (x2, y2), 2, (0, 255, 0))
            cv.line(img1, (x1, y1), (x2, y2), (0, 255, 0))
            cv.line(img2, (x1, y1), (x2, y2), (0, 255, 0))
        self.img_flip[]
        '''
        
        F, status = cv2.findFundamentalMat(mp1, mp2, cv2.FM_RANSAC, 5.0)
        status = status.ravel() != 0
        print '%d / %d' % (sum(status), len(status))
        mp1, mp2 = mp1[status], mp2[status]


        
        rectified_size = (800, 800)
        retval, H1, H2 = cv2.stereoRectifyUncalibrated(mp1.reshape(1, -1, 2), mp2.reshape(1, -1, 2), F, rectified_size)
        gH1 = np.dot(H1, img1_view)
        gH2 = np.dot(H2, img2_view)

        mp1 = cv2.perspectiveTransform(mp1.reshape(1, -1, 2), H1)
        mp2 = cv2.perspectiveTransform(mp2.reshape(1, -1, 2), H2)
        d = mp1[0,:,0]-mp2[0,:,0]

        def draw_vis(img, H, size):
            return cv2.warpPerspective(img, H, size)
        vis1 = draw_vis(self.frames[0].lods[0], gH1, rectified_size)
        vis2 = draw_vis(self.frames[1].lods[0], gH2, rectified_size)

        white = (255, 255, 255)
        for (x1, y1), (x2, y2) in np.int32(zip(mp1.reshape(-1, 2), mp2.reshape(-1, 2))):
            cv2.circle(vis1, (x1, y1), 2, white)
            cv2.circle(vis2, (x2, y2), 2, white)
            #cv2.line(vis1, (x1, y1), (x2, y2), white)
            #cv2.line(vis2, (x1, y1), (x2, y2), white)
        
        anaglyph = vis2.copy()
        anaglyph[..., 2] = vis1[..., 2]

        cv2.imshow('anaglyph', anaglyph)
        
        
        #e1 = cv2.canny(cv2.cvtColor(vis1, cv.CV_BGR2GRAY), 100, 200)
        #e2 = cv2.canny(cv2.cvtColor(vis2, cv.CV_BGR2GRAY), 100, 200)
        #edge_anaglyph = np.dstack([e2, e2, e1])
        #anaglyph = np.maximum(anaglyph, edge_anaglyph)
        
        print 'stereo matching...'
        disp = stereo.calc_disparity(vis1, vis2, d.min(), d.max())
        
        fnbase = '%02d_' % self.shot_idx

        print 'saving ply...'
        verts = np.zeros(disp.shape + (3,), np.float32)
        verts[...,1], verts[...,0] = np.ogrid[ :rectified_size[1], :rectified_size[0] ]
        verts[...,2] = disp*4
        verts *= 0.1
        write_ply(fnbase+'cloud.ply', verts, cv2.cvtColor(vis1, cv.CV_BGR2RGB))
        
        
        vis_disp = disp.copy()
        vis_disp -= d.min()
        vis_disp /= np.percentile(vis_disp, 99)
        vis_disp = np.uint8(np.clip(vis_disp, 0, 1)*255)
        
        #cv2.imshow('disp', vis_disp)
        #cv2.imshow('anaglyph', anaglyph)

        #cv2.imwrite(fnbase+'l.bmp', vis1)
        #cv2.imwrite(fnbase+'r.bmp', vis2)
        #cv2.imwrite(fnbase+'anaglyph.bmp', anaglyph)
        #cv2.imwrite(fnbase+'disp.bmp', vis_disp)
        #cv2.imwrite(fnbase+'small.bmp', self.cur_preview)
        
        self.shot_idx += 1


        
    def onkey(self, ch):
        pass

    def run(self):
        while True:
            ch = cv2.waitKey()
            if ch == 27:
                break
            self.onkey(ch)
        


        

if __name__ == '__main__':
    App().run()
