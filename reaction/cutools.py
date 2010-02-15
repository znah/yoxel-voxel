from ctypes import c_uint32, c_int32, c_float, Structure, addressof, sizeof
import numpy as np

import pycuda.driver as cu
import pycuda.gpuarray as ga
import pycuda.tools
from pycuda.compiler import SourceModule

def CU_PTR(ref_type):
    class cu_ptr(c_uint32):
        _ref_type_ = ref_type
        def __init__(self, d_ptr = 0):
            c_uint.__init__(self, int(d_ptr))
    return cu_ptr

ctype2name = {
  c_int32  : 'int', 
  c_uint32 : 'unsigned int',
  c_float  : 'float'}


def gen_struct(struct):
    def getcname(t):
        if t in ctype2name:
            return ctype2name[t]
        return t.__name__

    s = "struct %s\n{\n" % (struct.__name__,)

    for field, ctype in struct._fields_:
      if type(ctype).__name__ == 'ArrayType':
        s += "  %s %s[%d];\n" % (getcname(ctype._type_), field, ctype._length_)
      elif ctype.__name__ == 'cu_ptr':
        s += "  %s * %s;\n" % (getcname(ctype._ref_type_), field)
      else:
        s += "  %s %s;\n" % (getcname(ctype), field)
    s += "};\n"
    return s

def gen_code(ctype):
    if type(ctype).__name__ == 'StructType':
       return gen_struct(ctype)
    else:
       raise NotImplementedError;

def struct(name, *fields):
    class ttt(Structure):
        _fields_ = fields
    ttt.__name__ = name
    return ttt

def make_cu_vec(name, t, n):
    comp = ['x', 'y', 'z', 'w']
    fields = zip(comp, [t]*n)
    s = struct( name + str(n), *fields )
    s.dtype = np.dtype(fields)
    return s

def make_cu_vecs(name, t):
    return dict( [(name + str(i), make_cu_vec(name, t, i)) for i in xrange(1, 5)] )

cuda_vectors = {}
cuda_vectors.update( make_cu_vecs('int', c_int32) )
cuda_vectors.update( make_cu_vecs('uint', c_uint32) )
cuda_vectors.update( make_cu_vecs('float', c_float) )
globals().update(cuda_vectors)

range3i = struct('range3i', ('lo', int3), ('hi', int3))

cu_header = '#include "cutil_math.h"\n\n'
cu_header += gen_code(range3i)
cu_header += '''
__device__ bool inrange(range3i r, int3 p)
{
  if (p.x < r.lo.x || p.y < r.lo.y || p.z < r.lo.z)
    return false;
  if (p.x >= r.hi.x || p.y >= r.hi.y || p.z >= r.hi.z)
    return false;
  return true;
}

__device__ int getbid() 
{
  return blockIdx.x + blockIdx.y * gridDim.x;
}

__device__ int gettid() 
{
  return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
}

'''

if __name__ == '__main__':
    print gen_code(float4)

    Test = struct('Test', 
          ( 'p1', CU_PTR(int2) ),
          ( 'p2', c_float      ),
          ( 'p3', c_float * 3  ),
          ( 'p4', float3 * 2   ))
    print gen_code(Test)

    p = int3(1, 2, 3)
    print p.x, p.y, p.z
    print sizeof(p)
    print hex(addressof(p))

    print cu_header

    print int4.dtype



