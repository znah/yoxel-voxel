from __future__ import with_statement
from numpy import *
#import pyximport; pyximport.install()
#import cmesh

#cmesh.test()

class Pool:
    def __init__(self, dtype_):
        self.data = zeros((16,), dtype_)
        self.count = 0
    def extend(self, n = 1):
        data = self.data
        if self.count + n > len(data):
            capacity = max(2*len(data), self.count + n)
            self.data = resize(data, (capacity, ) + data.shape[1:])
        idx = self.count
        self.count += n
        return idx
    def push_back(self, v):
        i = self.extend(1)
        self[i] = v
        return i
    def set(self, a):
        self.data = a
        self.count = len(a)
    def __len__(self):
        return self.count
    def __getitem__(self, idx):
        return self.data[idx]
    def __setitem__(self, idx, v):
        self.data[idx] = v
    def __str__(self):
        return str(self.data[:self.count])
    def __repr__(self):
        return repr(self.data[:self.count])
       
class CoralMesh:
    def __init__(self):
        self.verts = Pool(dtype(float32) * 3)
        self.faces = Pool(dtype(uint32) * 3)
        
    def calc_normals(self):
        self.normals = normals = zeros_like(self.verts.data)
        faces = self.faces
        verts = self.verts
        p0 = verts[faces[:,0]]
        v1 = verts[faces[:,1]] - p0
        v2 = verts[faces[:,2]] - p0
        fn = cross(v1, v2)
        for i, face in enumerate(faces):
            n = fn[i]
            for v in face:
                normals[v] += n
        length = sqrt((normals * normals).sum(-1))
        normals /= length[:,newaxis]
        return normals
        
    def make_face(self, fid, verts):
        a, b, c = verts
        e1, e2, e3 = (a, b), (b, c), (c, a)
        self.edgeNext[e1] = e2
        self.edgeNext[e2] = e3
        self.edgeNext[e3] = e1
        self.edgeFace[e1] = fid
        self.edgeFace[e2] = fid
        self.edgeFace[e3] = fid
        self.faces[fid] = verts
        
    def build_edges(self):
        self.edgeNext = {}
        self.edgeFace = {}
        for fid, verts in enumerate(self.faces):
            self.make_face(fid, verts)
            
    def split_face(self, e, v):
        fid1 = self.edgeFace.pop(e)
        fid2 = self.faces.extend()
        a, b, c = e[1], self.edgeNext.pop(e)[1], e[0]
        self.make_face(fid1, (v, a, b))
        self.make_face(fid2, (v, b, c))
        
    def split_edge(self, e):
        c = 0.5 #0.3 + random.rand()*0.4
        pos = (c*self.verts[e[0]] + (1-c)*self.verts[e[1]])
        v = self.verts.push_back(pos)
        self.split_face(e, v)
        self.split_face((e[1], e[0]), v)
        
    def edge_len(self, e):
        dv = self.verts[e[0]] - self.verts[e[1]]
        return sqrt( (dv*dv).sum(-1) )
        
    def split_edges(self, threshold):
        to_split = [e for e in self.edgeFace 
            if e[0] < e[1] and self.edge_len(e) > threshold]
        for e in to_split:
            self.split_edge(e)
        return len(to_split)
        
    def del_edges(self, *es):
        for e in es:
            self.edgeFace.pop(e)
            self.edgeNext.pop(e)
        
    def flip_edge(self, e):
        re = (e[1], e[0])
        fid1 = self.edgeFace[e]
        fid2 = self.edgeFace[re]
        a, b, c, d = e[1], self.edgeNext[e][1], e[0], self.edgeNext[re][1]
        self.del_edges(e, re)
        self.make_face(fid1, (a, b, d))
        self.make_face(fid2, (b, c, d))
        
        
    def optimize(self):
        edges = self.edgeFace.keys()
        for e in edges:
            if e in self.edgeFace:
                self.flip_edge(e)
            
        #haveFlips = True
        #while haveFlips:
        #    haveFlips = False
        #for e in 
            
        

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
            
            self.mesh = mesh =  CoralMesh()
            mesh.verts.set(verts)
            mesh.faces.set(idxs)
            mesh.build_edges()

        def draw(self):
            glVertexPointer(3, GL_FLOAT, 0, self.mesh.verts.data)
            with glstate(GL_VERTEX_ARRAY):
                glDrawElements(GL_TRIANGLES, len(self.mesh.faces)*3, 
                    GL_UNSIGNED_INT, self.mesh.faces.data)
        
        growBtn = Button(label = "Grow")
        optimBtn = Button(label = "Optimize")
        view = View(Item( 'growBtn' ), Item('optimBtn'))
        def _growBtn_fired(self):
            #ns = self.mesh.calc_normals()
            self.mesh.verts[:] *= 1.1#+= 0.1 * ns
            while self.mesh.split_edges(0.5):
                pass
            print len(self.mesh.verts), len(self.mesh.faces)
            
        def _optimBtn_fired(self):
            self.mesh.optimize()

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
