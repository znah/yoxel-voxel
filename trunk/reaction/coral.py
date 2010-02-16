from zgl import *
from cutools import *
import os
import _coralmesh

class Diffusion:
    def __init__(self, sz = 256):
        
        code = cu_header + '''
          #line 10
          texture<uint4, 1> coralVoxels;
          texture<float, 3> srcTex;          
          
          const int SliceDepth   = 128;
          const int ChannelDepth = 32;
          
          __device__ void prepareChannel(uint bits, int3 &p, float *&dst, int stride)
          {
            for (uint i = 0; i < ChannelDepth; ++i)
            {
              float v = -1.0f;
              if ((bits & (1<<i)) == 0)
                v = tex3D(srcTex, p.x, p.y, p.z);
              *dst = v;
              dst += stride;
              ++p.z;
            }
          }
          
          extern "C"
          __global__ void PrepareDiffusion(int size, float * dst)
          {
            int3 p;
            p.x = threadIdx.x + blockIdx.x * blockDim.x;
            p.y = threadIdx.y + blockIdx.y * blockDim.y;
            p.z = 0;
            
            dst += p.x + p.y*size;
            int stride_z = size*size;
            
            while (p.z < size)
            {
              uint4 bits = tex1D(coralVoxels, p.x + p.y*size + p.z/SliceDepth*stride_z);
              prepareChannel(bits.x, p, dst, stride_z);
              prepareChannel(bits.y, p, dst, stride_z);
              prepareChannel(bits.z, p, dst, stride_z);
              prepareChannel(bits.w, p, dst, stride_z);
            }
          }
        
        '''
        self.mod = mod = SourceModule(code, include_dirs = [os.getcwd(), os.getcwd()+'/include'], no_extern_c = True)
        self.prepareFunc = mod.get_function("PrepareDiffusion")



class Coral:
    def __init__(self):
        self.gridSize = gridSize = 256
        self.coralliteSpacing = spacing = 1.8

        verts, idxs = load_obj('data/icosahedron.obj')
        verts *= spacing / 2.0
        verts += (gridSize/2, gridSize/2, spacing*2)
        self.mesh = _coralmesh.CoralMesh()
        for v in verts:
            mesh.add_vert(*v.tolist())
        for f in idxs:
            mesh.add_face(*f.tolist())
        mesh.update_normals()

        self.voxelizer = Voxelizer(gridSize)
        self.diffusion = accretive.Diffusion(gridSize)

    def grow(self):
        pass





        
        
        
        
if __name__ == '__main__':
    import pycuda.autoinit
    diff = Diffusion(256)
    
        
        
        