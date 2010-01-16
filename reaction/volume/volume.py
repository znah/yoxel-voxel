from __future__ import with_statement
import sys
sys.path.append("..")
from zgl import *

import pycuda.driver as cu
import pycuda.gpuarray as ga
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule


class CuSparseVolume:
    def __init__(self, brickSize = 8):
        self.brickSize = brickSize
        self.brickSize3 = brickSize**3
        self.capacity = 256
        self.d_bricks = ga.zeros((self.capacity,) + (brickSize,)*3, float32)
        self.brickMap = {}

    def allocBrick(self, pos):
        pos = tuple(pos)
        if pos in self.brickMap:
            return self.brickMap[pos]

        brickN = len(self.brickMap)
        if brickN >= self.capacity:
            self.capacity *= 2;
            d_new = ga.zeros((self.capacity, ) + (brickSize,)*3, float32)
            cu.memcpy_dtod(d_new, self.d_bricks, self.d_bricks.nbytes)
            self.d_bricks.gpudata.free()
            self.d_bricks = d_new

        self.brickMap[pos] = brickN
        return brickN

    def brickOfs(self, idx):
        return idx * self.brickSize3 * self.d_bricks.dtype.itemsize

    def __setitem__(self, pos, data):
        idx = self.allocBrick(pos)
        d_ptr = int(self.d_bricks.gpudata) + self.brickOfs(idx)
        data = ascontiguousarray(data, float32)
        cu.memcpy_htod(d_ptr, data)

    def __getitem__(self, pos):
        pos = tuple(pos)
        a = zeros((self.brickSize,)*3, self.d_bricks.dtype)
        if pos not in self.brickMap:
            return a
        idx = self.brickMap[pos]
        d_ptr = int(self.d_bricks.gpudata) + self.brickOfs(idx)
        cu.memcpy_dtoh(a, d_ptr)
        return a



        
'''
class App(ZglAppWX):


    def __init__(self):
        ZglAppWX.__init__(self, viewControl = FlyCamera())
    
    def display(self):
        pass
'''
if __name__ == '__main__':
    #App().run()
    vol = CuSparseVolume()

    a = arange(8**3, dtype = float32)
    a.shape = (8, 8, 8)

    vol[1, 2, 3] = a
    b = vol[1, 2, 3]
    print abs(a - b).max()


