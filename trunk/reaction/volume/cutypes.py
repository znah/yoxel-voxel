from ctypes import c_uint32, c_int32, c_float, Structure, addressof, sizeof

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
    return struct( name + str(n), *zip(comp, [t]*n) )

def make_cu_vecs(name, t):
    return dict( [(name + str(i), make_cu_vec(name, t, i)) for i in xrange(1, 5)] )

globals().update( make_cu_vecs('int', c_int32) )
globals().update( make_cu_vecs('uint', c_uint32) )
globals().update( make_cu_vecs('float', c_float) )

range3i = struct('range3i', ('lo', int3), ('hi', int3))

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



