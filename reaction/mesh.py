from __future__ import with_statement
from zgl import *
import _coralmesh


def create_box_mesh():
    verts, idxs, qidxs = create_box()
    
    mesh = _coralmesh.CoralMesh()
    for v in verts:
        mesh.add_vert(*v.tolist())
    for f in idxs:
        mesh.add_face(*f.tolist())
    mesh.update_normals()
    return mesh

def filterUnusedVerts(verts, faces):
    mark = zeros(len(verts), int32)
    mark[faces] = 1
    ofs = cumsum(mark)-1
    verts = verts[mark==1].copy()
    faces = ofs[faces]
    return verts, faces


class MeshTestApp(ZglAppWX):
    def __init__(self):
        ZglAppWX.__init__(self, viewControl = FlyCamera())

        self.viewControl.zNear = 0.01
        
        self.mesh = create_box_mesh()

        self.verts = self.mesh.get_positions()
        self.faces = self.mesh.get_faces()
        self.normals = self.mesh.get_normals()

    def draw(self):
        drawArrays(GL_TRIANGLES, verts = self.verts, indices = self.faces)

    def drawNormals(self):
        verts = self.verts
        normals = self.normals
        n = len(verts)
        v = zeros((2*n, 3), float32)
        v[::2] = verts
        glColor3f(1, 0, 0)
        v[1::2] = verts + normals * self.growStep
        drawArrays(GL_LINES, verts=v)
    
    growBtn = Button(label = "Grow")
    saveBtn = Button(label = "Save tmp.obj")
    view = View(Item('growBtn'), Item('saveBtn'), Item('growStep'))
    growStep = Float(0.1)

    def _growBtn_fired(self):
        n = self.mesh.get_vert_num()
        amounts = (random.rand(n)*self.growStep).astype(float32)

        self.mesh.grow(0.75, 1.5, amounts)
        self.verts = self.mesh.get_positions()
        self.faces = self.mesh.get_faces()
        self.normals = self.mesh.get_normals()

    #def _saveBtn_fired(self):
    #    self.mesh.save('tmp.obj')

    def OnKeyDown(self, evt):
        key = evt.GetKeyCode()
        if key == ord(' '):
            self._growBtn_fired()
        else:
            ZglAppWX.OnKeyDown(self, evt)


    def display(self):
        #self._growBtn_fired()

        clearGLBuffers()
        with ctx(self.viewControl.with_vp, glstate(GL_DEPTH_TEST, GL_DEPTH_CLAMP_NV, GL_CULL_FACE)):
            glColor3f(0.5, 0.5, 0.5)
            self.draw()

            glColor3f(1, 1, 1)
            with glstate(GL_POLYGON_OFFSET_LINE):
                glPolygonOffset(-1, -1)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                self.draw()
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            self.drawNormals()


if __name__ == '__main__':
    MeshTestApp().run()
    '''
    mesh = create_box_mesh()
    n = 300
    amounts = empty((1000,), float32)
    amounts[:] = 0.05
    times = zeros((n,))
    verts = zeros((n,), int32)
    for i in xrange(n):
        vn = mesh.get_vert_num()
        verts[i] = vn
        if len(amounts) < vn:
            amounts = resize(amounts, (vn*2,))
        mesh.grow(0.25, 1.0, amounts)
        if i % 100 == 0:
            print vn
        times[i] = clock()
    dt = diff(times)
    print times[-1] - times[0]
    save_obj('t.obj', mesh.get_positions(), mesh.get_faces())

    import pylab
    pylab.plot(dt / verts[1:])
    #pylab.plot(verts)
    pylab.show()
    '''