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
        
        self.block = block = (16, 16, 4)
        self.grid = (size / self.block[0], size / self.block[1] * size / self.block[2])
        self.cublock = (16, 4, 1)
        
        self.src = ga.zeros(shape, float32)
        self.dst = ga.zeros(shape, float32)
        
        self.Ctx = Ctx = struct('Ctx', 
            ( 'size'    , c_int32)
            ( 'stride_y', c_int32 )
            ( 'stride_z', c_int32 ))
            
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
        }

        const int3 blockSize = { {{v.block[0]}}, {{v.block[1]}}, {{v.block[2]}} };
        __shared__ float shared[3][{{v.block[1]}}+2][{{v.block[0]}}+2];


        __device__ int3 getBrickIdx(int3 gridSize)
        {
          int bid = getbid();
          int x = bid % gridSize.x; bid /= gridSize.x;
          int y = bid % gridSize.y; bid /= gridSize.y;
          int z = bid;
          return make_int3(x, y, z);
        }
          
        __device__ fetchSlice(int layer, int ofs, int tx, int ty)
        {
          shared[layer][ty][tx] = 

        }
        
        extern "C"
        __global__ void DiffusionShared(float * dst, int3 gridSize)
        {
          int3 p = getBrickIdx(gridSize);
          int tx = threadIdx.x;
          int ty = threadIdx.y;
          p.x = p.x * blockSize.x + tx;
          p.y = p.y * blockSize.y + ty;
          p.z = p.z * blockSize.z;
          int ofs = cell_ofs(p.x, p.y, p.z);

          for (int z = 0; z < blockSize; ++z)
          {
            if (z == 0)
            {
              shared[0][ty][tx] = tex1Dfetch(srcTex, ofs - stride_z);
              shared[1][ty][tx] = tex1Dfetch(srcTex, ofs);
              shared[2][ty][tx] = tex1Dfetch(srcTex, ofs + stride_z);
              if ()
            }
            else
            {
              shared[0][ty][tx] 
              


            }
            __syncthreads();


          }


          
       
          


        }
        
        ''').render(v = vars(), g = globals())
        self.mod = SourceModule(code, include_dirs = [os.getcwd(), os.getcwd()+'/include'], no_extern_c = True)
        ctx = Ctx(size = size)
        d_ctx = self.mod.get_global('ctx')
        cu.memcpy_htod(d_ctx[0], ctx)
        self.srcTexRef = self.mod.get_texref('srcTex')
        self.DiffKernel = self.mod.get_function('Diffusion')
        print self.DiffKernel.num_regs

    @with_( cuprofile("DiffusionStep") )
    def step(self, time_kernel = False):
        self.src.bind_to_texref(self.srcTexRef)
        t = self.DiffKernel(self.dst, block = self.block, grid = self.grid, time_kernel = time_kernel)
        self.flipBuffers()
        return t

    def flipBuffers(self):
        self.src, self.dst = self.dst, self.src

        
if __name__ == '__main__':
    class App(ZglAppWX):
        volumeRender = Instance(VolumeRenderer)

        def __init__(self):
            ZglAppWX.__init__(self, viewControl = FlyCamera())

            gridSize = 256
            
            a = zeros([gridSize]*3, float32)
            col = linspace(0.0, 1.0, gridSize).astype(float32)
            a[:] = col[...,newaxis, newaxis]
            
            sinks = (random.rand(10000, 3)*(gridSize, gridSize/2, gridSize/4)).astype(int32)
            for x, y, z in sinks:
                a[z, y, x] = Diffusion.SINK
            a[:, gridSize/2 + 1, :gridSize/2] = Diffusion.OBSTACLE
            
            self.diffusion = Diffusion(gridSize)
            self.diffusion.src.set(a)
                        
            volumeTex = Texture3D( size = [gridSize]*3, format = GL_LUMINANCE_FLOAT32_ATI )
            self.volumeRender = VolumeRenderer(volumeTex)
            self.step()

        def step(self):
            for i in xrange(50):
                self.diffusion.step()
                print '.',
            print
            a = self.diffusion.src.get()
            with self.volumeRender.volumeTex:
                glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, a.shape[0], a.shape[1], a.shape[2], GL_LUMINANCE, GL_FLOAT, a)

        def key_SPACE(self):
            self.step()

        def display(self):
            clearGLBuffers()
            
            with ctx(self.viewControl.with_vp):
                self.volumeRender.render()

    '''
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
    '''    
    import pycuda.autoinit
    App().run()
    

