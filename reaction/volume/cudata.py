from ctypes import *

def CU_PTR(ref_type):
    class cu_ptr(c_uint):
        _ref_type_ = ref_type
        def __init__(self, d_ptr):
            c_uint.__init__(self, d_ptr)
    return cu_ptr

def gen_struct(struct):
    ctype2name = {c_int32 : 'int', 
      c_uint32 : 'unsigned int',
      c_float : 'float'}
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
    s += "};\n\n"
    return s

def make_cu_vec(name, t, n):
    comp = ['x', 'y', 'z', 'w']
    class vec(Structure):
        _fields_ = [(comp[i], t) for i in xrange(n)]
    vec.__name__ = name + str(n)
    return vec

def make_cu_vecs(name, t):
    return dict( [(i, make_cu_vec(name, t, i)) for i in xrange(1, 5)] )

cu_intv = make_cu_vecs('int', c_int)
cu_uintv = make_cu_vecs('uint', c_uint)
cu_floatv = make_cu_vecs('float', c_float)
          
if __name__ == '__main__':
    code = ''


    for tt in [cu_intv, cu_uintv, cu_floatv]:
        for n in tt:
            code += gen_struct(tt[n])

    class Test(Structure):
        _fields_ = [('f1', cu_intv[3]*5), 
          ('f2', CU_PTR(cu_floatv[4]))]
    code += gen_struct(Test)

    print code
