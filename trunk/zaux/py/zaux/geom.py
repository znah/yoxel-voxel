import numpy as np

def solve_2x2(a, b, c, d, e, f):
    D =  a*d - b*c
    eps = 1e-8
    assert np.all( abs(D) > eps ) # TODO
    x = (e*d - b*f) / D
    y = (a*f - e*c) / D
    return x, y

def smult(a, b):
    return (a*b).sum(-1)

def intersect_lines(p1, d1, p2, d2):
    '''
      returns closest points
    '''
    a = smult(d1, d1)
    b = c = -smult(d1, d2)
    d = smult(d2, d2)
    dp = p2-p1
    e = smult(d1, dp)
    f = -smult(d2, dp)
    t1, t2 = solve_2x2(a, b, c, d, e, f)
    pc1 = p1 + d1*t1.reshape(-1,1)
    pc2 = p2 + d2*t2.reshape(-1,1)
    return pc1, pc2
