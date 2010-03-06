from zgl import *
from cutools import *
from volvis import VolumeRenderer
import os

class Diffusion:
    OBSTACLE = -1.0
    SINK = -2.0
    
    def __init__(self, size = 256):
        self.size = size
        shape = (size,)*3
        
        self.block = (8, 8, 4)
        self.grid = (size / self.block[0], size / self.block[1] * size / self.block[2])
        
        self.src = ga.zeros(shape, float32)
        self.dst = ga.zeros(shape, float32)
        
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
            else if (v == {{v.self.OBSTACLE}}f)
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
            if (self < 0 || z == 0 || z == ctx.size-1)
            {
              dst[ofs] = self;
              return;
            }

            const float c0 = 1.0f / 3.0f, c1 = 1.0f / 18.0f, c2 = 1.0f / 36.0f;
            float acc = c0 * self;
            float adj1 = 0.0f;
            adj1 += fetchNeib(self, x+1, y,   z  );
            adj1 += fetchNeib(self, x,   y+1, z  );
            adj1 += fetchNeib(self, x,   y,   z+1);
            adj1 += fetchNeib(self, x-1, y,   z  );
            adj1 += fetchNeib(self, x,   y-1, z  );
            adj1 += fetchNeib(self, x,   y,   z-1);
            acc += c1 * adj1;

            float adj2 = 0.0f;
            adj2 += fetchNeib(self, x+1, y+1, z  );
            adj2 += fetchNeib(self, x-1, y+1, z  );
            adj2 += fetchNeib(self, x+1, y-1, z  );
            adj2 += fetchNeib(self, x-1, y-1, z  );
            adj2 += fetchNeib(self, x+1, y, z+1  );
            adj2 += fetchNeib(self, x-1, y, z+1  );
            adj2 += fetchNeib(self, x+1, y, z-1  );
            adj2 += fetchNeib(self, x-1, y, z-1  );
            adj2 += fetchNeib(self, x, y+1, z+1  );
            adj2 += fetchNeib(self, x, y-1, z+1  );
            adj2 += fetchNeib(self, x, y+1, z-1  );
            adj2 += fetchNeib(self, x, y-1, z-1  );
            acc += c2 * adj2;

            dst[ofs] = acc;
        }''').render(v = vars(), g = globals())
        self.mod = SourceModule(code, include_dirs = [os.getcwd(), os.getcwd()+'/include'], no_extern_c = True)
        ctx = Ctx(size = size)
        d_ctx = self.mod.get_global('ctx')
        cu.memcpy_htod(d_ctx[0], ctx)
        self.srcTexRef = self.mod.get_texref('srcTex')
        self.DiffKernel = self.mod.get_function('Diffusion')
        
    def step(self, time_kernel = False):
        with cuprofile("DiffusionStep"):
            self.src.bind_to_texref(self.srcTexRef)
            t = self.DiffKernel(self.dst, block = self.block, grid = self.grid, time_kernel = time_kernel)
            self.flipBuffers()
        return t

    def flipBuffers(self):
        self.src, self.dst = self.dst, self.src
        
if __name__ == '__main__':
    class App(ZglAppWX):
        def __init__(self):
            ZglAppWX.__init__(self, viewControl = FlyCamera())
            self.diffusion = Diffusion(64)
            a = zeros((64,)*3, float32)
            
            a[9:,19:,29:][:3,:3,:3] = Diffusion.SINK
            a[11, 20, 30] = Diffusion.OBSTACLE
            a[10, 20, 30] = 1.0
            a[10, 21, 30] = 0.0
            
            self.diffusion.src.set(a)
            print self.diffusion.step(time_kernel = True)
            a = self.diffusion.src.get()
            print a[9:,19:,29:][:3,:3,:3]
        
    import pycuda.autoinit
    App()#.run()
    

