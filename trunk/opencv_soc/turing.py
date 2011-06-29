import numpy as np
import cv2, cv

w, h = 1024, 768

a = np.zeros((h, w, 1), np.float32)
cv2.randu(a, np.array([0]), np.array([1]))
a.shape = (h, w)

def process_scale(a_lods, lod):
    a1 = a_lods[lod]
    a2 = cv2.pyrUp(a_lods[lod+1])
    d = a1-a2
    for i in xrange(lod):
        d = cv2.pyrUp(d)
    v = cv2.gaussianBlur(d*d, (3, 3), 0)
    return np.sign(d), v
    
out = cv2.VideoWriter('turing.avi', cv.CV_FOURCC(*'DIB '), 30.0, (w, h), False)

for i in xrange(1000):
    if i % 30 == 0:
        print i
    a_lods = [a]
    for i in xrange(7):
        a_lods.append(cv2.pyrDown(a_lods[-1])) 
    ms, vs = [], []
    for i in xrange(1, 7):
        m, v = process_scale(a_lods, i)
        ms.append(m)
        vs.append(v)
    mi = np.argmin(vs, 0)
    a += np.choose(mi, ms) * 0.025
    a = (a-a.min()) / a.ptp()

    out.write(a)
    cv2.imshow('a', a)
    if cv2.waitKey(5) == 27:
        break
else:
    print 'done'
    cv2.waitKey()