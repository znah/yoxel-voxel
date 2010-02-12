from __future__ import with_statement
from numpy import *
import _coralmesh

class Pool:
    def __init__(self, dtype_):
        self.data = zeros((16,), dtype_)
        self.count = 0
        self.view = self.data[:self.count]
    def extend(self, n = 1):
        data = self.data
        if self.count + n > len(data):
            capacity = max(2*len(data), self.count + n)
            self.data = resize(data, (capacity, ) + data.shape[1:])
        idx = self.count
        self.count += n
        self.view = self.data[:self.count]
        return idx
    def push_back(self, v):
        i = self.extend(1)
        self[i] = v
        return i
    def pop_back(self):
        self.count -= 1
        self.view = self.data[:self.count]
    def set(self, a):
        self.data = a
        self.count = len(a)
        self.view = self.data[:self.count]
    def __len__(self):
        return self.count
    def __getitem__(self, idx):
        return self.view[idx]
    def __setitem__(self, idx, v):
        self.view[idx] = v
    def __str__(self):
        return str(self.view)
    def __repr__(self):
        return repr(self.view)
    def __iter__(self):
        return iter(self.view)


def flip(e):
    return (e[1], e[0])
        
class CoralMesh:
    def __init__(self):
        self.verts = Pool(dtype(float32) * 3)
        self.faces = Pool(dtype(uint32) * 3)
        
    def calc_normals(self):
        self.normals = normals = zeros_like(self.verts.view)
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
        length[length < 1e-5] = 1.0
        normals /= length[:,newaxis]
        return normals
        
    def make_face(self, fid, vertIds):
        a, b, c = vertIds
        e1, e2, e3 = (a, b), (b, c), (c, a)
        self.edgeNext[e1] = e2
        self.edgeNext[e2] = e3
        self.edgeNext[e3] = e1
        self.edgeFace[e1] = fid
        self.edgeFace[e2] = fid
        self.edgeFace[e3] = fid
        self.faces[fid] = vertIds
        
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
        self.split_face(flip(e), v)
        
    def edge_len(self, e):
        dv = self.verts[e[0]] - self.verts[e[1]]
        return sqrt( (dv*dv).sum(-1) )
        
    def split_edges(self, threshold):
        to_split = [e for e in self.edgeFace 
            if e[0] < e[1] and self.edge_len(e) > threshold]
        for e in to_split:
            self.split_edge(e)
        if len(to_split) > 0:
            print "split:", len(to_split)
        return len(to_split)

    def del_edges(self, *es):
        for e in es:
            del self.edgeFace[e]
            del self.edgeNext[e]

    def wipe_face(self, fid):
        a, b, c = self.faces[fid]
        self.del_edges( (a, b), (b, c), (c, a) )
        if fid != len(self.faces)-1:
            last = self.faces[-1]
            self.make_face(fid, last)
        self.faces.pop_back()

    def wipe_vert(self, edge):
        '''
        wipe all faces adjacent to first vertex of edge
        '''
        border = []
        edge = flip(edge)
        while edge in self.edgeNext:
            border.append(edge[0])
            next = flip( self.edgeNext[edge] )
            self.wipe_face( self.edgeFace[edge] )
            edge = next
        return [border[0]] + border[1:][::-1]
    
    def fill_hole(self, border):
        for i in xrange(2, len(border)):
            fid = self.faces.extend()
            f = (border[i-1], border[i], border[0])
            self.make_face(fid, f)
    
    def shrink_edge(self, e):
        c = 0.5
        self.verts[e[1]] = (c*self.verts[e[0]] + (1-c)*self.verts[e[1]])
        holeBorder = flip(self.edgeNext[e])
        border = self.wipe_vert(e)
        self.fill_hole(border)

    def shrink_edges(self, threshold):
        to_shrink = [e for e in self.edgeFace 
            if e[0] < e[1] and self.edge_len(e) < threshold]
        count = 0
        for e in to_shrink:
            if e in self.edgeFace:
                self.shrink_edge(e)
                count += 1
        if count > 0:
            print "shrink:", count
        return count

    def save(self, fn):
        f = file(fn, 'w')
        f.write("# verts: %d\n# faces: %d\n\n" % (len(self.verts), len(self.faces)))
        for v in self.verts:
            f.write("v %f %f %f\n" % tuple(v))
        for face in self.faces:
            f.write("f %d %d %d\n" % tuple(face+1))
        f.close()


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
            mesh.calc_normals()

            self.mesh2 = mesh2 = _coralmesh.CoralMesh()
            for v in verts:
                mesh2.add_vert(*v.tolist())
            for f in idxs:
                mesh2.add_face(*f.tolist())
            mesh2.update_normals()
            self.verts = self.mesh2.get_positions()
            self.faces = self.mesh2.get_faces()
            self.normals = self.mesh2.get_normals()



        def draw(self):
            #verts = self.mesh.verts.view
            #faces = self.mesh.faces.view
            verts = self.verts
            faces = self.faces

            glVertexPointer(3, GL_FLOAT, 0, verts)
            with glstate(GL_VERTEX_ARRAY):
                glDrawElements(GL_TRIANGLES, len(faces)*3, 
                    GL_UNSIGNED_INT, faces)

        def drawNormals(self):
            verts = self.verts
            normals = self.normals

            n = len(verts)
            v = zeros((2*n, 3), float32)
            v[::2] = verts
            glColor3f(1, 0, 0)
            v[1::2] = verts + normals * self.growStep
            glVertexPointer(3, GL_FLOAT, 0, v)
            with glstate(GL_VERTEX_ARRAY):
                glDrawArrays(GL_LINES, 0, 2*n)
            
        
        growBtn = Button(label = "Grow")
        saveBtn = Button(label = "Save tmp.obj")
        view = View(Item('growBtn'), Item('saveBtn'))
        growStep = Float(0.05)

        def _growBtn_fired(self):
            n = self.mesh2.get_vert_num()
            amounts = zeros((n,), float32)
            amounts[:] = self.growStep

            self.mesh2.grow(0.25, 1.0, amounts)
            self.verts = self.mesh2.get_positions()
            self.faces = self.mesh2.get_faces()
            self.normals = self.mesh2.get_normals()
            print len(self.verts), len(self.faces)

            '''
            self.mesh.verts[:] += self.mesh.normals * self.growStep
            self.mesh.verts[:,2] += self.growStep * sin(self.mesh.verts[:,1])
            while self.mesh.shrink_edges(0.25):
                pass
            while self.mesh.split_edges(1.0):
                pass
            self.mesh.calc_normals()
            print len(self.mesh.verts), len(self.mesh.faces)
            '''
            
        def _saveBtn_fired(self):
            self.mesh.save('tmp.obj')
    
        def OnKeyDown(self, evt):
            key = evt.GetKeyCode()
            if key == ord(' '):
                self._growBtn_fired()
            else:
                ZglAppWX.OnKeyDown(self, evt)


        def display(self):
            clearGLBuffers()
            with ctx(self.viewControl.with_vp, glstate(GL_DEPTH_TEST, GL_DEPTH_CLAMP_NV)):
                glColor3f(0.5, 0.5, 0.5)
                self.draw()

                glColor3f(1, 1, 1)
                with glstate(GL_POLYGON_OFFSET_LINE):
                    glPolygonOffset(-1, -1)
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                    self.draw()
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                self.drawNormals()

    
    App().run()
