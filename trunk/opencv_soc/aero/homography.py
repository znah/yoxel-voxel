import numpy as np
from numpy import sqrt, dot, cross

def decomposeH(H):
    H = H.copy()
    u, s, v = np.linalg.svd(H)
    H /= s[1]

    U, (s1, s2, s3), V = np.linalg.svd(dot(H.T, H))
    if np.linalg.det(V) < 0.0:
        V = -V
    v1, v2, v3 = V
    u1 = (v1*sqrt(1-s3) + v3*sqrt(s1 - 1))/sqrt(s1 - s3)
    u2 = (v1*sqrt(1-s3) - v3*sqrt(s1 - 1))/sqrt(s1 - s3)

    sol = []
    for u in [u1, u2]:
        n = cross(v2,u)
        U = np.array([v2, u, n]).T
        Hv2, Hu = dot(H,v2), dot(H,u)
        W = np.array([Hv2, Hu, cross(Hv2, Hu)]).T
        if n[2] < 0.0:
            n = -n
        R = dot(W, U.T)
        td = dot((H - R), n)
        sol.append((R, td, n))


    return sol

if __name__ == '__main__':
    import cv2

    R = cv2.Rodrigues(np.float64([0, 0, 0]))[0]
    n = np.float64([0.0, 0.0, 1])
    n /= np.linalg.norm(n)
    t = np.float64([0, 0, 1])
    H = R + np.outer(t, n)
    print n

    for dR, dtd, dn in decomposeH(H):
        #print R-dR, t-dtd, n-dn
        print dR
        print dtd
        print dn
        print
        





'''

h, w = 7216, 5412
sensor_width = 49.0
focal_length = 50.0

f = w * focal_length / sensor_width
K = float64([[ f, 0, 0.5*w],
             [ 0, f, 0.5*h],
             [ 0, 0,     1]])
Ki = linalg.inv(K)

'''

'''
 if det(u) < 0 u = -u; end;
 
 s1 = s(1,1); s2 = s(2,2); s3 = s(3,3);
 v1 = u(:,1); v2 = u(:,2); v3 = u(:,3);
 u1 = (v1*sqrt(1-s3) + v3*sqrt(s1 -1))/sqrt(s1 - s3);
 u2 = (v1*sqrt(1-s3) - v3*sqrt(s1 -1))/sqrt(s1 - s3);
 
 
 U1 = [v2, u1, skew(v2)*u1];
 U2 = [v2, u2, skew(v2)*u2];
 W1 = [H*v2, H*u1, skew(H*v2)*H*u1];
 W2 = [H*v2, H*u2, skew(H*v2)*H*u2];
 
 N1 = skew(v2)*u1;
 N2 = skew(v2)*u2;
 
 Sol(:,:,1) = [W1*U1', (H - W1*U1')*N1, N1];
 Sol(:,:,2) = [W2*U2', (H - W2*U2')*N2, N2];
 Sol(:,:,3) = [W1*U1', -(H - W1*U1')*N1, -N1];
 Sol(:,:,4) = [W2*U2', -(H - W2*U2')*N2, -N2];
'''
