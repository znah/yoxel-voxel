from zgl import *
from cutools import *
import os
import _coralmesh
import pycuda.gl as cuda_gl
from diffusion import Diffusion
from voxelizer import Voxelizer




class Coral:
    def __init__(self):
        self.gridSize = gridSize = 128
        self.coralliteSpacing = 1.8
        self.mergeDist = 0.75 * self.coralliteSpacing
        self.splitDist = 1.5  * self.coralliteSpacing
        self.mouthDist = 3.0
        self.growCoef = 0.5
        self.diffuseStepNum = 40

        self.initMesh()
        self.voxelizer = Voxelizer(gridSize)
        self.diffusion = Diffusion(gridSize)
        a = zeros([gridSize]*3, float32)
        col = linspace(0.0, 1.0, gridSize).astype(float32)
        a[:] = col[...,newaxis, newaxis]
        self.diffusion.src.set(a)
        
        self.setupKernels()
        self.getMeshArrays()
        
    def initMesh(self):
        spacing = self.coralliteSpacing
        verts, idxs = load_obj('data/icosahedron.obj')
        verts *= spacing / 2.0
        #erts, idxs = load_obj('data/shpere_642.obj')
        #verts *= spacing / 0.14
        
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
          
          #line 47
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
                v = max( 0.0f, tex1Dfetch(srcTex, ofs) );
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
            
          __device__ bool ingrid(int3 p, int gridSize)
          {
            return p.x >= 0 && p.x < gridSize && 
                   p.y >= 0 && p.y < gridSize &&
                   p.z >= 0 && p.z < gridSize;
          }
          
          extern "C"
          __global__ void MarkSinks(int gridSize, float * grid, float mark, int sinkNum, const float3 * pos)
          {
            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx >= sinkNum)
              return;
            float3 p = pos[idx];
            int3 c = make_int3(p);
            if (!ingrid(c, gridSize))
              return;
            grid[c.x + (c.y + c.z * gridSize) * gridSize] = mark;
          }
          
          extern "C"
          __global__ void FetchSinks(int gridSize, int sinkNum, const float3 * pos, float * dst)
          {
            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx >= sinkNum)
              return;
            float3 p = pos[idx];
            int3 c = make_int3(p);
            if (!ingrid(c, gridSize))
              return;
            dst[idx] = tex1Dfetch(srcTex, c.x + (c.y + c.z * gridSize) * gridSize);
          }
        ''').render(v=vars(), g = globals())
        mod = SourceModule(code, include_dirs = [os.getcwd(), os.getcwd()+'/include'], no_extern_c = True)
        voxelBitsTex = mod.get_texref('voxelBits')
        voxelBitsTex.set_format(cu.array_format.UNSIGNED_INT32, 4)
        voxelBitsTex.set_flags(cu.TRSF_READ_AS_INTEGER)
        srcTex = mod.get_texref('srcTex')
        
        PrepareDiffusionVolume = mod.get_function("PrepareDiffusionVolume")
        def func(d_voxelBitsPtr, bitsSize):
            block = (16, 16, 1)
            grid = (self.gridSize/block[0], self.gridSize/block[1])
            self.diffusion.src.bind_to_texref(srcTex)
            voxelBitsTex.set_address(d_voxelBitsPtr, bitsSize)
            PrepareDiffusionVolume(int32(self.gridSize), self.diffusion.dst, 
                block = block, grid = grid, 
                texrefs = [voxelBitsTex, srcTex])
            self.diffusion.flipBuffers()
        self.PrepareDiffusionVolume = func
        
        MarkSinks = mod.get_function("MarkSinks")
        def func(d_sinkPos, mark):
            block = (256, 1, 1)
            n = len(d_sinkPos)
            MarkSinks( int32(self.gridSize), self.diffusion.src, float32(mark), int32(n), d_sinkPos,
              block = block, grid = ((n + block[0] - 1) /  block[0], 1) )
        self.MarkSinks = func
        
        FetchSinks = mod.get_function("FetchSinks")
        def func(d_sinkPos, d_absorb):
            block = (256, 1, 1)
            self.diffusion.src.bind_to_texref(srcTex)
            n = len(d_sinkPos)
            FetchSinks( int32(self.gridSize), int32(n), d_sinkPos, d_absorb,
              block = block, grid = ((n + block[0] - 1) /  block[0], 1), 
              texrefs = [srcTex])
        self.FetchSinks = func
        
        
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
        self.PrepareDiffusionVolume(gl2cudaMap.device_ptr(), gl2cudaMap.size())
        gl2cudaMap.unmap()
        
        
        d_sinks = ga.to_gpu(self.positions + self.normals * self.mouthDist)
        self.MarkSinks(d_sinks, Diffusion.SINK)
        for i in xrange(self.diffuseStepNum):
            self.diffusion.step()
            if (i+1) % 10 == 0:
                print ga.sum(abs(self.diffusion.src - self.diffusion.dst)  )
        self.MarkSinks(d_sinks, 0.0)
        self.diffusion.step()
        d_absorb = ga.zeros(len(d_sinks), float32)
        self.FetchSinks(d_sinks, d_absorb)
        absorb = d_absorb.get()
        absorb /= absorb.max()
        self.mesh.grow(self.mergeDist, self.splitDist, absorb*self.growCoef)
        self.getMeshArrays()
        
if __name__ == '__main__':
    class App(ZglAppWX):
        def __init__(self):
            ZglAppWX.__init__(self, viewControl = FlyCamera())
            import pycuda.gl.autoinit
            
            self.viewControl.speed = 10
            
            self.coral = Coral()
            self.coral.grow()
            
        def OnKeyDown(self, evt):
            key = evt.GetKeyCode()
            if key == ord(' '):
                self.coral.grow()
            if key == ord('1'):
                save_obj("t.obj", self.coral.positions, self.coral.faces)
            if key == ord('2'):
                for i in xrange(10):
                    self.coral.grow()
            else:
                ZglAppWX.OnKeyDown(self, evt)
                
        def display(self):
            clearGLBuffers()
            with ctx(self.viewControl.with_vp, glstate(GL_DEPTH_TEST, GL_DEPTH_CLAMP_NV)):
                glColor3f(0.5, 0.5, 0.5)
                drawArrays(GL_TRIANGLES, verts = self.coral.positions, indices = self.coral.faces)

                glColor3f(1, 1, 1)
                with glstate(GL_POLYGON_OFFSET_LINE):
                    glPolygonOffset(-1, -1)
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                    drawArrays(GL_TRIANGLES, verts = self.coral.positions, indices = self.coral.faces)
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                
        
    
    App().run()
