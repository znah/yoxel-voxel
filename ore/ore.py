import sys
#from numpy import *
from _ore import *


def p3i(p):
    p = map(int, p)
    return point_3i(p[0], p[1], p[2])

def p3f(p):
    p = map(float, p)
    return point_3f(p[0], p[1], p[2])
