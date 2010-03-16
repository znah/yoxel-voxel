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
            
        code = Template('''
          {{g.cu_header}}
          {{g.gen_code(v.Ctx)}}
          //__constant__ Ctx ctx;
        
          #line 44
          texture<float, 1> srcTex;

          #define OBSTACLE {{v.self.OBSTACLE}}f 
          #define SINK     {{v.self.SINK}}f
          
          const int BLOCK_DIM_X = {{v.block[0]}};
          const int BLOCK_DIM_Y = {{v.block[1]}};
          const int BLOCK_DIM_Z = {{v.block[2]}};
          const int BLOCK_SIZE  = BLOCK_DIM_X*BLOCK_DIM_Y*BLOCK_DIM_Z;

          const int SHRD_DIM_X = {{v.block[0]}} + 2;
          const int SHRD_DIM_Y = {{v.block[1]}} + 2;
          const int SHRD_DIM_Z = {{v.block[2]}} + 2;
          const int SHRD_SIZE  = SHRD_DIM_X*SHRD_DIM_Y*SHRD_DIM_Z;

          const int GRID_DIM_X = {{v.grid[0]}};
          const int GRID_DIM_Y = {{v.grid[1]}};
          const int GRID_DIM_Z = {{v.grid[2]}};

          const int VOL_DIM_X = GRID_DIM_X * BLOCK_DIM_X;
          const int VOL_DIM_Y = GRID_DIM_Y * BLOCK_DIM_Y;
          const int VOL_DIM_Z = GRID_DIM_Z * BLOCK_DIM_Z;
          const int VOL_SIZE  = VOL_DIM_X*VOL_DIM_Y*VOL_DIM_Z;

          __device__ float getflux(float self, float neib)
          {
            if (neib == SINK)
              neib = 0.0f;
            if (neib == OBSTACLE)
              neib = self;
            return neib;
          }

          
          __shared__ float smem      [SHRD_SIZE];
          __constant__ int fetch_reqs[SHRD_SIZE];

          extern "C"
          __global__ void Diffusion(const float * src, float * dst)
          {
            int bx = blockIdx.x;
            int by = blockIdx.y % GRID_DIM_Y;
            int bz = blockIdx.y / GRID_DIM_Y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int tz = threadIdx.z;
            int x0 = bx * BLOCK_DIM_X;
            int y0 = by * BLOCK_DIM_Y;
            int z0 = bz * BLOCK_DIM_Z;
            int blockBaseOfs = x0 + y0*VOL_DIM_X + z0*VOL_DIM_X*VOL_DIM_Y;
            int x = x0 + tx;
            int y = y0 + ty;
            int z = z0 + tz;

            int tid = tx + ty*BLOCK_DIM_X + tz*BLOCK_DIM_X*BLOCK_DIM_Y;
            for (int i = tid; i < SHRD_SIZE; i += BLOCK_SIZE)
            {
              int req = fetch_reqs[i];
              int ofs = blockBaseOfs + req;
              smem[i] = tex1Dfetch(srcTex, ofs);
            }

            __syncthreads();

            int ofs = x + y*VOL_DIM_X + z*VOL_DIM_X*VOL_DIM_Y;
            
            const int SX = 1, SY = SHRD_DIM_X, SZ = SHRD_DIM_X * SHRD_DIM_Y;
            int sofs = (tx+1) + (ty+1)*SY + (tz+1)*SZ;
            
            float self = smem[sofs];
            if (self < 0 || z == 0 || z == VOL_DIM_Z-1)
            {
              dst[ofs] = self;
              return;
            }

            const float c0 = 1.0f / 3.0f, c1 = 1.0f / 18.0f, c2 = 1.0f / 36.0f;
            float acc = c0 * self;

            float adj1 = 0.0f;
            adj1 += getflux(self, smem[sofs + SX] );
            adj1 += getflux(self, smem[sofs - SX] );
            adj1 += getflux(self, smem[sofs + SY] );
            adj1 += getflux(self, smem[sofs - SY] );
            adj1 += getflux(self, smem[sofs + SZ] );
            adj1 += getflux(self, smem[sofs - SZ] );
            acc += c1 * adj1;

            float adj2 = 0.0f;
            adj2 += getflux(self, smem[sofs + SX + SY] );
            adj2 += getflux(self, smem[sofs - SX + SY] );
            adj2 += getflux(self, smem[sofs + SX - SY] );
            adj2 += getflux(self, smem[sofs - SX - SY] );

            adj2 += getflux(self, smem[sofs + SX + SZ] );
            adj2 += getflux(self, smem[sofs - SX + SZ] );
            adj2 += getflux(self, smem[sofs + SX - SZ] );
            adj2 += getflux(self, smem[sofs - SX - SZ] );

            adj2 += getflux(self, smem[sofs + SY + SZ] );
            adj2 += getflux(self, smem[sofs - SY + SZ] );
            adj2 += getflux(self, smem[sofs + SY - SZ] );
            adj2 += getflux(self, smem[sofs - SY - SZ] );
            acc += c2 * adj2;

            dst[ofs] = acc;
        }
        ''').render(v = vars(), g = globals())
        mod = SourceModule(code, include_dirs = [os.getcwd(), os.getcwd()+'/include'], no_extern_c = True) #options = ['--maxrregcount=16'])
        #d_ctx        = mod.get_global('ctx')
        d_fetch_reqs = mod.get_global('fetch_reqs')
        #cu.memcpy_htod(d_ctx[0], ctx)
        cu.memcpy_htod(d_fetch_reqs[0], fetch_reqs)
        srcTexRef = mod.get_texref('srcTex')
        DiffKernel = mod.get_function('Diffusion')

        @with_( cuprofile("DiffusionStep") )
        def step():
            self.src.bind_to_texref(srcTexRef)
            DiffKernel(self.src, self.dst, block = block, grid = cu_grid)
            self.flipBuffers()
        self.step = step
    
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
            
            sinks = (random.rand(10000, 3)*(gridSize, gridSize/2, gridSize/2)).astype(int32)
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
    

