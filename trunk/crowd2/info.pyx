cimport numpy as np

cdef f(np.npy_int * a):
    print *a

cdef np.npy_int i = 5
f(&i)