import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

cdef extern from "path_field.h":
    cdef cppclass array_2d_ref[T]:
        array_2d_ref()
        array_2d_ref(int[2], T*)
    cdef cppclass float2:
        pass
    ctypedef array_2d_ref[float] Grid2Dref
    ctypedef array_2d_ref[float2] V2Grid2Dref
    void calc_distmap(Grid2Dref & obstmap, Grid2Dref & distmap, V2Grid2Dref & pathmap)

def _calc_distmap(dens):
    h, w = dens.shape
    distmap = np.zeros((h, w), np.float32)
    pathmap = np.zeros((h, w, 2), np.float32)

    cdef np.ndarray[float, ndim=2] dens_view = dens
    cdef np.ndarray[float, ndim=2] dist_view = distmap
    cdef np.ndarray[float, ndim=3] path_view = pathmap

    cdef Grid2Dref dens_ref = Grid2Dref([h, w], <float*>dens_view.data)
    cdef Grid2Dref dist_ref = Grid2Dref([h, w], <float*>dist_view.data)
    cdef V2Grid2Dref path_ref = V2Grid2Dref([h, w], <float2*>path_view.data)

    calc_distmap(dens_ref, dist_ref, path_ref)
    return distmap, pathmap

        

