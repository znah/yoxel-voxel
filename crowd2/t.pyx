import numpy as np
cimport numpy as np
from libcpp.vector cimport vector


cdef extern from "RVOServer.h":
    void test123(float*, int[2])

cdef extern from "path_field.h":
    cdef cppclass Grid2Dref:
         Grid2Dref()
         Grid2Dref(float*, int[2])
    cdef cppclass V2Grid2Dref:  
         Grid2Dref()
         #Grid2Dref(float*, int[3])
    void calc_distmap(Grid2Dref & obstmap, Grid2Dref & distmap, V2Grid2Dref & pathmap)


def test1():                                                              
    cdef np.ndarray[float, ndim=2] A = np.float32(np.arange(25)).reshape(5, 5)

    '''
    cdef vector[int] vec
    vec.push_back(5)
    test123(vec)
    for i in xrange(vec.size()):
        print vec[i]
    '''
    test123(<float*>A.data, A.shape)


    #cdef Grid2Dref grid = Grid2Dref(<float*>A.data, [10, 10])

    #del grid
    #print 'asdfasf'

        

