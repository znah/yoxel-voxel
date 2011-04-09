'''
  Coherence-enhancing filtering example

  inspired by  
    Joachim Weickert "Coherence-Enhancing Shock Filters"
    http://www.mia.uni-saarland.de/Publications/weickert-dagm03.pdf
'''

from numpy import *
import cv


def coference_filter(img, sigma = 11, blend = 0.5, iter_n = 4):
    img = asarray(cv.GetMat(img)).copy()
    h, w = img.shape[:2]

    gray  = zeros((h, w), uint8)
    eigen_buf = zeros((h, w*6), float32)
    gxx = zeros((h, w), float32)
    gxy = zeros((h, w), float32)
    gyy = zeros((h, w), float32)
    dil = zeros_like(img)
    ero = zeros_like(img)

    st = cv.CreateStructuringElementEx(3, 3, 1, 1, cv.CV_SHAPE_RECT)
    for i in xrange(iter_n):
        print i,
        
        cv.CvtColor(img, gray, cv.CV_BGR2GRAY)
        cv.CornerEigenValsAndVecs(gray, eigen_buf, 11)
        eigen = eigen_buf.reshape(h, w, 3, 2)  # [[e1, e2], v1, v2]
        x, y = eigen[:,:,1,0], eigen[:,:,1,1]
        
        cv.Sobel(gray, gxx, 2, 0, sigma)
        cv.Sobel(gray, gxy, 1, 1, sigma)
        cv.Sobel(gray, gyy, 0, 2, sigma)
        gvv = x*x*gxx + 2*x*y*gxy + y*y*gyy
        m = gvv < 0

        cv.Erode(img, ero, st)
        cv.Dilate(img, dil, st)
        img1 = ero
        img1[m] = dil[m]
        img[:] = uint8(img*(1.0 - blend) + img1*blend)

    print 'done'

    return img

    
if __name__ == '__main__':
    import sys
    try: fn = sys.argv[1]
    except: fn = 'images/thai.jpg'

    src = cv.LoadImage(fn)

    def nothing(*argv):
        pass

    def update():
        sigma = cv.GetTrackbarPos('sigma', 'dst')*2+1
        blend = cv.GetTrackbarPos('blend', 'dst') / 10.0
        print 'sigma: %d   blend_coef: %f' % (sigma, blend)
        dst = coference_filter(src, sigma=sigma, blend = blend)
        cv.ShowImage('dst', dst)

    cv.NamedWindow('dst')
    cv.CreateTrackbar('sigma', 'dst', 11, 15, nothing)
    cv.CreateTrackbar('blend', 'dst', 5, 10, nothing)


    print 'Press SPACE to update the image\n'

    cv.ShowImage('src', src)
    update()
    while True:
        ch = cv.WaitKey(0)
        if ch == ord(' '):
            update()
        if ch == 27:
            break
