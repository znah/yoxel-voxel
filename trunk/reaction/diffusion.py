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
        
        self.src = ga.zeros(shape, float32)
        self.dst = ga.zeros(shape, float32)
        
        block = (16, 4, 4)
        grid, cu_grid = make_grid3d(shape, block)
        
        Ctx = struct('Ctx', 
            ( 'size'    , c_int32 ),
            ( 'strideY' , c_int32 ),
            ( 'strideZ' , c_int32 ),
            ( 'gridSize', int3    ))
        ctx = Ctx()
        ctx.size = size
        ctx.strideY = size
        ctx.strideZ = size * size
        ctx.gridSize = int3(*grid)

        domain = array(block) + 2
        dz, dy, dx = ogrid[-1:domain[2]-1, -1:domain[1]-1, -1:domain[0]-1]
        fetch_reqs = (dz*size*size + dy*size + dx).astype(int32)
        fetch_reqs = fetch_reqs.ravel()
        print fetch_reqs

            
        code = Template('''
          {{g.cu_header}}
          {{g.gen_code(v.Ctx)}}
          __constant__ Ctx ctx;
        
          #line 37
          texture<float, 1> srcTex;
          
          __device__ float getflux(float self, float neib)
          {
            if (neib >= 0)
              return neib;
            else if (neib == {{v.self.OBSTACLE}}f)
              return self;
            else 
              return 0.0f;
          }

          const int BLOCK_DIM_X = {{v.block[0]}};
          const int BLOCK_DIM_Y = {{v.block[1]}};
          const int BLOCK_DIM_Z = {{v.block[2]}};
          const int BLOCK_SIZE = BLOCK_DIM_X*BLOCK_DIM_Y*BLOCK_DIM_Z;

          const int DOM_DIM_X = {{v.block[0]}} + 2;
          const int DOM_DIM_Y = {{v.block[1]}} + 2;
          const int DOM_DIM_Z = {{v.block[2]}} + 2;
          const int DOM_SIZE = DOM_DIM_X*DOM_DIM_Y*DOM_DIM_Z;

          __shared__ float smem      [DOM_SIZE];
          #define SMEM(x, y, z)   smem[ (x)+1 + ((y)+1)*DOM_DIM_X + ((z)+1)*DOM_DIM_X*DOM_DIM_Y ]
          __constant__ int fetch_reqs[DOM_SIZE];
          const int REQ_OFS_MASK = 0x8fffffff;
          const int REQ_WRAP_X = 1<<30;
          const int REQ_WRAP_Y = 1<<29;
          const int REQ_WRAP_Z = 1<<28;


          extern "C"
          __global__ void Diffusion(float * dst)
          {
            int bx = blockIdx.x;
            int by = blockIdx.y % ctx.gridSize.y;
            int bz = blockIdx.y / ctx.gridSize.y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int tz = threadIdx.z;
            int x0 = bx * BLOCK_DIM_X;
            int y0 = by * BLOCK_DIM_Y;
            int z0 = bz * BLOCK_DIM_Z;
            int blockBaseOfs = x0 + y0 * ctx.strideY + z0 * ctx.strideZ;
            int x = x0 + tx;
            int y = y0 + ty;
            int z = z0 + tz;

            int tid = tx + (ty + tz * BLOCK_DIM_Y) * BLOCK_DIM_X;
            for (int i = tid; i < DOM_SIZE; i += BLOCK_SIZE)
            {
              int req = fetch_reqs[i];
              int ofs = blockBaseOfs + (req/* & REQ_OFS_MASK*/);
              if (req & REQ_WRAP_X)
              {
                if (x == 0) ofs += ctx.strideY;
                if (x == ctx.size-1) ofs -= ctx.strideY;
              }
              if (req & REQ_WRAP_Y)
              {
                if (y == 0) ofs += ctx.strideZ;
                if (y == ctx.size-1) ofs -= ctx.strideZ;
              }
              if (req & REQ_WRAP_Z)
              {
                if (z == 0) ofs += ctx.strideZ * ctx.size;
                if (z == ctx.size-1) ofs -= ctx.strideZ * ctx.size;
              }
              smem[i] = tex1Dfetch(srcTex, ofs);
            }
            __syncthreads();

            int ofs = x + y * ctx.strideY + z * ctx.strideZ;
            
            float self = SMEM(0, 0, 0);
            if (self < 0 || z == 0 || z == ctx.size-1)
            {
              dst[ofs] = self;
              return;
            }


            const float c0 = 1.0f / 3.0f, c1 = 1.0f / 18.0f, c2 = 1.0f / 36.0f;
            float acc = c0 * self;

            float adj1 = 0.0f;
            adj1 += getflux(self, SMEM( tx+1, ty , tz ) );
            adj1 += getflux(self, SMEM( tx , ty+1, tz ) );
            adj1 += getflux(self, SMEM( tx , ty , tz+1) );
            adj1 += getflux(self, SMEM( tx-1, ty , tz ) );
            adj1 += getflux(self, SMEM( tx , ty-1, tz ) );
            adj1 += getflux(self, SMEM( tx , ty , tz-1) );
            acc += c1 * adj1;

            float adj2 = 0.0f;
            adj2 += getflux(self, SMEM( tx+1, ty+1,  tz) );
            adj2 += getflux(self, SMEM( tx-1, ty+1,  tz) );
            adj2 += getflux(self, SMEM( tx+1, ty-1,  tz) );
            adj2 += getflux(self, SMEM( tx-1, ty-1,  tz) );
            adj2 += getflux(self, SMEM( tx+1,  ty, tz+1) );
            adj2 += getflux(self, SMEM( tx-1,  ty, tz+1) );
            adj2 += getflux(self, SMEM( tx+1,  ty, tz-1) );
            adj2 += getflux(self, SMEM( tx-1,  ty, tz-1) );
            adj2 += getflux(self, SMEM(  tx, ty+1, tz+1) );
            adj2 += getflux(self, SMEM(  tx, ty-1, tz+1) );
            adj2 += getflux(self, SMEM(  tx, ty+1, tz-1) );
            adj2 += getflux(self, SMEM(  tx, ty-1, tz-1) );
            acc += c2 * adj2;

            dst[ofs] = acc;
        }
        ''').render(v = vars(), g = globals())
        mod = SourceModule(code, include_dirs = [os.getcwd(), os.getcwd()+'/include'], no_extern_c = True) #options = ['--maxrregcount=16'])
        d_ctx        = mod.get_global('ctx')
        d_fetch_reqs = mod.get_global('fetch_reqs')
        cu.memcpy_htod(d_ctx[0], ctx)
        cu.memcpy_htod(d_fetch_reqs[0], fetch_reqs)
        srcTexRef = mod.get_texref('srcTex')
        DiffKernel = mod.get_function('Diffusion')
        print DiffKernel.num_regs

        @with_( cuprofile("DiffusionStep") )
        def step():
            self.src.bind_to_texref(srcTexRef)
            DiffKernel(self.dst, block = block, grid = cu_grid)
            self.src, self.dst = self.dst, self.src

        self.step = step

        
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
    

