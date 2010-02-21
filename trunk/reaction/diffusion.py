from zgl import *
from cutools import *
from volvis import VolumeRenderer

class Diffusion:
    def __init__(self, size = 256):
        self.size = size
        shape = (size,)*3
        
        self.src = ga.zeros(shape, float32)
        self.dst = ga.zeros(shape, float32)
        
        neib = indices((3, 3, 3)).T.reshape(-1, 3) - 1
        code = Template('''
          {{v.cu_header}}        
        
          #line 15
          texture<float, 1> srcTex;
          
          __device__ float fetch(int x, int y, int z, int size)
          {
            x = (x + size) % size;
            y = (y + size) % size;
            z = (z + size) % size;
            return tex1Dfetch(srcTex, x + (y + z * size) * size);
          }
          
          __global__ void Diffusion(int size, float * dst)
          {
            int bx = blockIdx.x;
            int gridSizeY = (size / blockDim.y);
            int by = blockIdx.y % gridSizeY;
            int bz = blockIdx.y / gridSizeY;
            
            int x = threadIdx.x + bx * blockDim.x;
            int y = threadIdx.y + by * blockDim.y;
            int z = threadIdx.z + bz * blockDim.z;
            
            {% for dz, dy, dx in v.neib %}
            {% adj = abs(dx) + abs(dy) + abs(dz) %}
            {{adj}} {{dy}} {{dz}}
            {% endfor %}
            
            float acc = 0.0f;
            {% for dx, dy, dz in v.neib %}\
              acc += {{w}} * fetch(x + {{dx}}, y + {{dy}}, z + {{dz}}, size);
            {% endfor %}
            
            return;
          
          
        ''').render(v = vars(), abs = abs)
        print code
        
        
        
if __name__ == '__main__':
    class App(ZglAppWX):
        def __init__(self):
            ZglAppWX.__init__(self, viewControl = FlyCamera())
            self.diffusion = Diffusion(64)
        
        
    import pycuda.autoinit
    App()#.run()
    

