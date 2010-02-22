from zgl import *
from cutools import *
from volvis import VolumeRenderer
import os

class Diffusion:
    def __init__(self, size = 256):
        self.size = size
        shape = (size,)*3
        
        self.block = (8, 8, 4)
        self.grid = (size / self.block[0], size / self.block[1] * size / self.block[2])
        
        self.src = ga.zeros(shape, float32)
        self.dst = ga.zeros(shape, float32)
        
        neibs = []
        weights = [1.0 / 3.0, 1.0 / 18.0, 1.0 / 36.0, 0.0]
        for z, y, x in ndindex(3, 3, 3):
            x, y, z = x-1, y-1, z-1
            adj = abs(x) + abs(y) + abs(z)
            if adj in [1, 2]:
                neibs.append((x, y, z, weights[adj]))
                
        self.Ctx = Ctx = struct('Ctx', 
            ( 'size', c_int32))

        code = Template('''
          {{g.cu_header}}
          {{g.gen_code(v.Ctx)}}
          __constant__ Ctx ctx;
        
          #line 30
          texture<float, 1> srcTex;
          
          __device__ uint cell_ofs(int x, int y, int z)
          {
            return x + (y + z * ctx.size) * ctx.size;
          }
          
          __device__ float fetchNeib(float self, int x, int y, int z)
          {
            const int size = ctx.size;
            x = (x + size) % size;
            y = (y + size) % size;
            z = (z + size) % size;
            float v = tex1Dfetch(srcTex, cell_ofs(x, y, z));
            if (v >= 0)
              return v;
            else if (v == -1.0f)
              return self;
            else 
              return 0.0f;
          }
          
          extern "C"
          __global__ void Diffusion(float * dst)
          {
            int bx = blockIdx.x;
            int gridSizeY = (ctx.size / blockDim.y);
            int by = blockIdx.y % gridSizeY;
            int bz = blockIdx.y / gridSizeY;
            
            int x = threadIdx.x + bx * blockDim.x;
            int y = threadIdx.y + by * blockDim.y;
            int z = threadIdx.z + bz * blockDim.z;
            uint ofs = cell_ofs(x, y, z);
            
            float self = tex1Dfetch(srcTex, ofs);
            float acc = self / 3.0f;\
            {% for dx, dy, dz, w in v.neibs %}
              acc += {{w}}f * fetchNeib(self, x + ({{dx}}), y + ({{dy}}), z + ({{dz}}));\
            {% endfor %}
            dst[ofs] = acc;
        }''').render(v = vars(), g = globals())
        self.mod = SourceModule(code, include_dirs = [os.getcwd(), os.getcwd()+'/include'], no_extern_c = True)
        ctx = Ctx(size = size)
        d_ctx = self.mod.get_global('ctx')
        cu.memcpy_htod(d_ctx[0], ctx)
        self.srcTexRef = self.mod.get_texref('srcTex')
        self.DiffKernel = self.mod.get_function('Diffusion')
        
    def step(self):
        self.src.bind_to_texref(self.srcTexRef)
        self.DiffKernel(self.dst, block = self.block, grid = self.grid)
        self.src, self.dst = self.dst, self.src
        
if __name__ == '__main__':
    class App(ZglAppWX):
        def __init__(self):
            ZglAppWX.__init__(self, viewControl = FlyCamera())
            self.diffusion = Diffusion(64)
            a = zeros((64,)*3, float32)
            a[10, 20, 30] = 1.0
            self.diffusion.src.set(a)
            self.diffusion.step()
            a = self.diffusion.src.get()
            print a[9:,19:,29:][:3,:3,:3]
        
    import pycuda.autoinit
    App()#.run()
    

