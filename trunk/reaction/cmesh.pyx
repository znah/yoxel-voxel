import numpy as np
cimport numpy as np

cdef extern from "vector.h":
    ctypedef struct intvec "std::vector<unsigned int>":
        void (* push_back)(int elem)
    intvec intvec_factory "std::vector<unsigned int>"(int len)

def test():
    print "test"
    