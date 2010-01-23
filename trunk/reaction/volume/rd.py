from volume import *


vol = CuSparseVolume()

code = cu_header + vol.header + '''
  #line 151
  extern "C" 
  __global__ void TestFill()
'''
