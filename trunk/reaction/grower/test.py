from _grower import *
from numpy import *
from time import clock

import sys
sys.path.append('c:/dev/reaction')
#import zgl


grow = LapGrow()
grow.SetExponent(5.0)

t = clock()
for i in xrange(50000):
    if i % 1000 == 0:
        print i
    grow.GrowParticle()     

print clock() - t

a = grow.FetchBlack();
lo, hi = a.min(0), a.max(0)
fld = zeros(hi-lo+1, int32)
for i, p in enumerate(a):
    p -= lo
    fld[p[0], p[1]] = 1

import pylab
pylab.gray()
pylab.imshow(fld)
pylab.show()

#raw_input("press enter to exit")

