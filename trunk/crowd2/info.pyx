cimport numpy as np

cdef f(np.int * a):
    print a[0]

cdef np.npy_int i = 5
f(&i)