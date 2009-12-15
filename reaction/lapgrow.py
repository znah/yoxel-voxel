from numpy import *


def neib(p):
    (x, y) = p
    return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

p0 = (0, 0)
black = set( [p0] )
n0 = neib(p0)
grey  = dict(zip(n0, [0.0]*len(n0)))
eps = 1e-6

for i in xrange(2000):
    sites = grey.keys()
    phi   = array(grey.values(), float32)
    
    lo, hi = phi.min(), phi.max()
    if hi - lo > eps:
        p = (phi - lo) / (hi - lo)
    else:
        p = ones(len(phi), float32)
    p **= 3.0
    p /= p.sum()

    p = cumsum(p)               
    newBlackIdx  = searchsorted(p, random.rand())
    newBlack = sites[newBlackIdx]
    
    xy = array(sites, float32)
    dx = ravel(xy[:,0]) - newBlack[0]
    dy = ravel(xy[:,1]) - newBlack[1]
    r = sqrt(dx*dx + dy*dy)
    phi += (1.0 - 1.0 / r)

    grey = dict(zip(sites, phi))
    grey.pop(newBlack)
    black.add(newBlack)

    newGrey = neib(newBlack)
    newGrey = [p for p in newGrey if p not in black]
    newGrey = [p for p in newGrey if p not in grey]

    xy = array(list(black), float32)
    for p in newGrey:
        d = xy - p
        x, y = d[:,0], d[:,1]
        r = sqrt(x*x + y*y)
        phi = 1.0 - (1.0/r)
        totalPhi = sum(phi)
        grey[p] = totalPhi



#print list(black)


import pylab


a = array(list(black))
x, y = a[:,0], a[:,1]

pylab.plot(x, y, '.')
pylab.show()
    


    
    
    
    




