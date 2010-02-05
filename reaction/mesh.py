from __future__ import with_statement
from numpy import *
#import pyximport; pyximport.install()
#import cmesh

#cmesh.test()

class Pool:
    def __init__(self, dtype_):
        self.data = zeros((16,), dtype_)
        self.count = 0
    def push_back(self, v):
        assert self.count <= len(self.data)
        if self.count == len(self.data):
            shape = list(data.shape)
            shape[0] *= 2
            self.data = resize(self.data, shape)
        idx = self.count
        self.data[idx] = v
        self.count += 1
        return idx
    def set(self, a):
        self.data = a
        self.count = len(a)
    def __len__(self):
        return self.count
    def __getitem__(self, idx):
        return self.data[idx]
    def __setitem__(self, idx, v):
        self.data[idx] = v
        
edge_t = dtype([
    ("nextEdge", int32),
    ("startVert", int32),
    ("face", int32)])


class CoralMesh:
    def __init__(self):
        self.verts = Pool(dtype(float32) * 3)
        self.faces = Pool(dtype(uint32) * 3)
     
    def init_edges(self):
        self.edges = Pool(edge_t)
        #for faceId, face in enumarate()

        
if __name__ == '__main__':
    from zgl import *
    
    class App(ZglAppWX):
        def __init__(self):
            ZglAppWX.__init__(self, viewControl = FlyCamera())
            
            verts = indices((2, 2, 2), float32).T.reshape(-1,3)
            idxs = [ 0, 2, 3, 1,
                     0, 1, 5, 4, 
                     4, 5, 7, 6,
                     1, 3, 7, 5,
                     0, 4, 6, 2,
                     2, 6, 7, 3]
            idxs = array(idxs, uint32).reshape(-1,4)
            idxs = idxs[:,(0, 1, 2, 0, 2, 3)]  # quads -> triangles
            idxs = idxs.reshape((-1,3))
            
            self.mesh = CoralMesh()
            self.mesh.verts.set(verts)
            self.mesh.faces.set(idxs)

        def draw(self):
            glVertexPointer(3, GL_FLOAT, 0, self.mesh.verts.data)
            with glstate(GL_VERTEX_ARRAY):
                glDrawElements(GL_TRIANGLES, len(self.mesh.faces)*3, 
                    GL_UNSIGNED_INT, self.mesh.faces.data)
    
        def display(self):
            clearGLBuffers()
            with ctx(self.viewControl.with_vp, glstate(GL_DEPTH_TEST)):
                glColor3f(0.5, 0.5, 0.5)
                self.draw()

                glColor3f(1, 1, 1)
                with glstate(GL_POLYGON_OFFSET_LINE):
                    glPolygonOffset(-1, -1)
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                    self.draw()
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)



    
    App().run()

    
    
    
    
    

    
    