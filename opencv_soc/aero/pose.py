import numpy as np
from numpy import sqrt, dot, cross


def makeK(img_size, focal_length, sensor_width):
    w, h = img_size
    f = w * focal_length / sensor_width
    K = np.float64([[ f, 0, 0.5*w],
                    [ 0, f, 0.5*h],
                    [ 0, 0,     1]])
    return K

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

def decomposeE(E):
    U, D, V = np.linalg.svd(E)
    W = np.float64( [[0, -1, 0], [ 1, 0, 0], [0, 0, 1]] )
    Z = np.float64( [[0,  1, 0], [-1, 0, 0], [0, 0, 0]] )

    S = dot( dot(U, Z), U.T )
    R1 = dot(dot( U, W), V.T )
    R2 = dot(dot( U, W.T), V )
    t1 =  U[:,2]
    t2 = -U[:,2]
    return [(R1, t1), (R2, t2)]

'''
%% essential matrix decomposition
[U,D,V] = svd(E);

W = [0 -1 0; 1 0 0; 0 0 1];
Z = [0  1 0;-1 0 0; 0 0 0];

%% translaton vector (skew-symmetry matrix)
S = U*Z*U';

%% two possible rotation matrices
R1 = U*W*V';
R2 = U*W'*V';

%% two possible translation vectors
t1 = U(:,3);
t2 = -U(:,3);
'''
    

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

w, h = 7216, 5412
sensor_width = 49.0
focal_length = 50.0

f = w * focal_length / sensor_width
K = float64([[ f, 0, 0.5*w],
             [ 0, f, 0.5*h],
             [ 0, 0,     1]])
Ki = linalg.inv(K)

'''
