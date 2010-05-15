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
        
        code = Template('''
          {{g.cu_header}}
        
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

          const int VOL_DIM_X = {{v.size}};
          const int VOL_DIM_Y = {{v.size}};
          const int VOL_DIM_Z = {{v.size}};
          const int VOL_SIZE  = VOL_DIM_X*VOL_DIM_Y*VOL_DIM_Z;

          __device__ float getflux(float self, float neib)
          {
            if (neib == SINK)
              neib = 0.0f;
            if (neib == OBSTACLE)
              neib = self;
            return neib;
          }
          
        const int TILE_DIM_X = 8;
        const int TILE_DIM_Y = 16;

        extern "C"
        __global__ void Diffusion(const float * src, float * dst, bool saturateMode)
        {
           __shared__ float smem[3][TILE_DIM_Y + 2][TILE_DIM_X + 2];

           int tx = threadIdx.x;
           int ty = threadIdx.y;
           if (tx >= TILE_DIM_X + 2 || ty >= TILE_DIM_Y + 2)
             return;

           int x = (tx-1) + blockIdx.x * TILE_DIM_X;
           int y = (ty-1) + blockIdx.y * TILE_DIM_Y;

           bool valid = tx >= 1 && ty >= 1 && tx <= TILE_DIM_X && ty <= TILE_DIM_Y;

           if (x < 0)           x += VOL_DIM_X;
           if (y < 0)           y += VOL_DIM_Y;
           if (x > VOL_DIM_X-1) x -= VOL_DIM_X;
           if (y > VOL_DIM_Y-1) y -= VOL_DIM_Y;

           int ofs = x + y * VOL_DIM_X;
           const int stride_z = VOL_DIM_X * VOL_DIM_Y;
           smem[1][ty][tx] = src[ofs];
           smem[2][ty][tx] = smem[1][ty][tx];

           const int max_z = VOL_DIM_Z-1;
           for (int z = 0; z <= max_z; ++z, ofs += stride_z)
           {
             smem[0][ty][tx] = smem[1][ty][tx];
             smem[1][ty][tx] = smem[2][ty][tx];
             if (z < max_z)
               smem[2][ty][tx] = src[ofs + stride_z];
             __syncthreads();

             if (!valid)
               continue;
             float self = smem[1][ty][tx];
             if (self == OBSTACLE)
             {
               dst[ofs] = self;
               continue;
             }

             if (!saturateMode && (self == SINK || z == 0 || z == max_z))
             {
               dst[ofs] = self;
               continue;
             }
             self = max(self, 0.0);

             const float c0 = 1.0f / 3.0f, c1 = 1.0f / 18.0f, c2 = 1.0f / 36.0f;
             float acc = c0 * self;

             float adj1 = 0.0f;
             adj1 += getflux(self, smem[1][ty][tx+1] );
             adj1 += getflux(self, smem[1][ty][tx-1] );
             adj1 += getflux(self, smem[1][ty+1][tx] );
             adj1 += getflux(self, smem[1][ty-1][tx] );
             adj1 += getflux(self, smem[0][ty][tx] );
             adj1 += getflux(self, smem[2][ty][tx] );
             acc += c1 * adj1;

             float adj2 = 0.0f;
             adj2 += getflux(self, smem[1][ty+1][tx+1] );
             adj2 += getflux(self, smem[1][ty+1][tx-1] );
             adj2 += getflux(self, smem[1][ty-1][tx+1] );
             adj2 += getflux(self, smem[1][ty-1][tx-1] );

             adj2 += getflux(self, smem[0][ty][tx+1] );
             adj2 += getflux(self, smem[0][ty][tx-1] );
             adj2 += getflux(self, smem[2][ty][tx+1] );
             adj2 += getflux(self, smem[2][ty][tx-1] );

             adj2 += getflux(self, smem[0][ty+1][tx] );
             adj2 += getflux(self, smem[0][ty-1][tx] );
             adj2 += getflux(self, smem[2][ty+1][tx] );
             adj2 += getflux(self, smem[2][ty-1][tx] );

             acc += c2 * adj2;

             dst[ofs] = acc;
           }
        }

        ''').render(v = vars(), g = globals())
        mod = SourceModule(code, include_dirs = [os.getcwd(), os.getcwd()+'/include'], no_extern_c = True, keep=True) #options = ['--maxrregcount=16'])
        srcTexRef = mod.get_texref('srcTex')
        
        DiffKernel = mod.get_function('Diffusion')
        print "reg: %d,  lmem: %d " % (DiffKernel.num_regs, DiffKernel.local_size_bytes)
        block2 = (8+2, 16+2, 1)
        grid2 = (size / 8, size / 16)

        @with_( cuprofile("DiffusionStep") )
        def step(saturate = False):
            self.src.bind_to_texref(srcTexRef)
            DiffKernel(self.src, self.dst, int32(saturate), block = block2, grid = grid2, texrefs = [srcTexRef])
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
            a[:128] = 1.0
            #col = linspace(0.0, 1.0, gridSize).astype(float32)
            #a[:] = col[...,newaxis, newaxis]
            
            #sinks = (random.rand(10000, 3)*(gridSize, gridSize/2, gridSize/2)).astype(int32)
            #for x, y, z in sinks:
            #    a[z, y, x] = Diffusion.SINK
            #a[:, gridSize/2 + 1, :gridSize/2] = Diffusion.OBSTACLE
            
            self.diffusion = Diffusion(gridSize)
            self.diffusion.src.set(a)
                        
            volumeTex = Texture3D( size = [gridSize]*3, format = GL_LUMINANCE_FLOAT32_ATI )
            self.volumeRender = VolumeRenderer(volumeTex)
            a = self.diffusion.src.get()
            with self.volumeRender.volumeTex:
                glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, a.shape[0], a.shape[1], a.shape[2], GL_LUMINANCE, GL_FLOAT, a)
            #self.step()

        def step(self):
            for i in xrange(1):
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
            print self.diffusion.step(saturate=True)
            a = self.diffusion.src.get()
            print a[9:,19:,29:][:3,:3,:3]
    '''    
    import pycuda.autoinit
    App().run()
    

