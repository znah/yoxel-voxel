#cython: boundscheck=False
#cython: wraparound=False

cimport cython
import numpy as np
cimport numpy as np

cdef packed struct float2:
    float x, y
cdef float2 add(float2 a, float2 b):
    return float2(a.x+b.x, a.y+b.y)
cdef float2 sub(float2 a, float2 b):
    return float2(a.x-b.x, a.y-b.y)
cdef float2 mul(float2 a, float c):
    return float2(a.x*c, a.y*c)

cdef float2 sample2d(np.float32_t[:,:,:] a, int x, int y):
    return float2(a[y, x, 0], a[y, x, 1])
cdef float2 sample2d_interp(np.float32_t[:,:,:] a, float2 p):
    cdef int xi = <int>p.x, yi = <int>p.y
    cdef float dx = p.x-xi, dy = p.y-yi
    cdef float2 v00, v10, v01, v11, v0, v1
    v00 = sample2d(a, xi, yi)
    v10 = sample2d(a, xi+1, yi)
    v01 = sample2d(a, xi, yi+1)
    v11 = sample2d(a, xi+1, yi+1)
    v0 = add(v00, mul(sub(v10, v00), dx)) 
    v1 = add(v01, mul(sub(v11, v01), dx)) 
    return add(v0, mul(sub(v1, v0), dy)) 
  

cdef class VOModelBase:
    cdef public np.float32_t time_step
    cdef public np.float32_t grid_step
    cdef public np.float32_t[:,:] pos
    cdef public np.float32_t[:,:,:] pathmap

    def step(self):
        cdef int i
        cdef float2 p, dp
        for i in xrange(len(self.pos)):
            p = float2(self.pos[i,0], self.pos[i,1])
            p = mul(p, self.grid_step)
            dp = sample2d_interp(self.pathmap, p)
            dp = mul(dp, self.time_step)
            p = add(p, dp)
            self.pos[i,0], self.pos[i,1] = p.x, p.y
