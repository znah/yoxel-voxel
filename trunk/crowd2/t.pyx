def test():
    cdef double s = 0.0
    cdef int i
    for i in xrange(100000):
        s += <double>i*i
    print s
    