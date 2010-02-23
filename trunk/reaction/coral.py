from zgl import *
from cutools import *
import os
import _coralmesh
import pycuda.gl as cuda_gl
from diffusion import Diffusion
from voxelizer import Voxelizer




class Coral:
    def __init__(self):
        self.gridSize = gridSize = 256
        self.coralliteSpacing = 1.8

        self.initMesh()
        self.voxelizer = Voxelizer(gridSize)
        self.diffusion = Diffusion(gridSize)
        a = zeros([gridSize]*3, float32)
        col = linspace(0.0, 1.0, gridSize).astype(float32)
        a[:] = col[...,newaxis, newaxis]
        a[10:, 128:, 100:][:10,:10,:10] = -2
        self.diffusion.src.set(a)
        
        self.setupKernels()
        self.getMeshArrays()
        
    def initMesh(self):
        spacing = self.coralliteSpacing
        verts, idxs = load_obj('data/icosahedron.obj')
        verts *= spacing / 2.0
        verts += (self.gridSize/2, self.gridSize/2, spacing*2)
        self.mesh = mesh = _coralmesh.CoralMesh()
        for v in verts:
            mesh.add_vert(*v.tolist())
        for f in idxs:
            mesh.add_face(*f.tolist())
        mesh.update_normals()
        
    def setupKernels(self):
        self.gl2cudaBuf = cuda_gl.BufferObject(self.voxelizer.dumpToPBO().handle)
        
        code = Template('''
          {{g.cu_header}}
          
          #line 44
          texture<uint4, 1> voxelBits;
          texture<float, 1> srcTex;
          
          const int SliceDepth   = 128;
          const int ChannelDepth = 32;
          
          __device__ void prepareChannel(
            uint bits, 
            float * dst, 
            int ofs, int stride)
          {
            for (uint i = 0; i < ChannelDepth; ++i)
            {
              float v = {{ v.self.diffusion.OBSTACLE }}f;
              if ((bits & (1<<i)) == 0)
                v = tex1Dfetch(srcTex, ofs); // max( 0.0f, tex1Dfetch(srcTex, ofs) );
              dst[ofs] = v;
              ofs += stride;
            }
          }
          
          extern "C"
          __global__ void PrepareDiffusionVolume(int size, float * dst)
          {
            int3 p;
            p.x = threadIdx.x + blockIdx.x * blockDim.x;
            p.y = threadIdx.y + blockIdx.y * blockDim.y;
            p.z = 0;
            
            int ofs = p.x + p.y*size;
            int stride_z = size*size;
            int stride_ch = stride_z * ChannelDepth;
            
            while (p.z < size)
            {
              uint4 bits = tex1Dfetch(voxelBits, p.x + p.y*size + p.z/SliceDepth*stride_z);
              prepareChannel(bits.x, dst, ofs, stride_z); ofs += stride_ch;
              prepareChannel(bits.y, dst, ofs, stride_z); ofs += stride_ch;
              prepareChannel(bits.z, dst, ofs, stride_z); ofs += stride_ch;
              prepareChannel(bits.w, dst, ofs, stride_z); ofs += stride_ch;
              p.z += SliceDepth;
            }
          }
        
        ''').render(v=vars(), g = globals())
        mod = SourceModule(code, include_dirs = [os.getcwd(), os.getcwd()+'/include'], no_extern_c = True)
        self.voxelBitsTex = mod.get_texref('voxelBits')
        self.voxelBitsTex.set_format(cu.array_format.UNSIGNED_INT32, 4)
        self.voxelBitsTex.set_flags(cu.TRSF_READ_AS_INTEGER)
        srcTex = mod.get_texref('srcTex')
        
        func = mod.get_function("PrepareDiffusionVolume")
        block = (16, 16, 1)
        grid = (self.gridSize/block[0], self.gridSize/block[1])
        def PrepareDiffusionVolume():
            self.diffusion.src.bind_to_texref(srcTex)
            func(int32(self.gridSize), self.diffusion.dst, block = block, grid = grid, 
                texrefs = [self.voxelBitsTex, srcTex])
            self.diffusion.flipBuffers()
        self.PrepareDiffusionVolume = PrepareDiffusionVolume
        
    def getMeshArrays(self):
        self.positions = self.mesh.get_positions()
        self.normals = self.mesh.get_normals()
        self.faces = self.mesh.get_faces()
        
    def grow(self):
        with self.voxelizer:
            clearGLBuffers()
            s = 1.0 / self.gridSize
            glScale(s, s, s)
            drawArrays(GL_TRIANGLES, verts = self.positions, indices = self.faces)
        self.voxelizer.dumpToPBO()
        gl2cudaMap = self.gl2cudaBuf.map()
        self.voxelBitsTex.set_address(gl2cudaMap.device_ptr(), gl2cudaMap.size())
        self.PrepareDiffusionVolume()
        gl2cudaMap.unmap()
        
        for i in xrange(100):
            self.diffusion.step()
            print ga.sum(abs(self.diffusion.src - self.diffusion.dst)  )
        
        
        a = self.diffusion.src.get()
        save("a", a[:,self.gridSize/2])
        
if __name__ == '__main__':
    class App(ZglAppWX):
        def __init__(self):
            ZglAppWX.__init__(self, viewControl = FlyCamera())
            import pycuda.gl.autoinit
            
            self.coral = Coral()
            self.coral.grow()
    
    App()#.run()
