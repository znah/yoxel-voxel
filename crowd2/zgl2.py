from __future__ import division
import numpy as np
import pyglet
import pyglet.gl as gl

def V(*v):
    return np.float64(v)

class OrthoView(object):
    def __init__(self, window):
        self.size = None
        self.center = V(0.0, 0.0)
        self.extent_x = 1.0
        self.anim_extent_x = self.extent_x
        self.anim_zoom_center = None
        
        window.push_handlers(self)
        pyglet.clock.schedule(self.update)

    def on_resize(self, w, h):
        self.size = (max(w, 1), max(h, 1))
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & pyglet.window.mouse.LEFT:
            self.center -= self.get_pixel_extent()*(dx, dy)
            return True
    def on_mouse_scroll(self, x, y, dx, dy):
        self.anim_zoom_center = self.scr2wld(x, y)
        self.anim_extent_x *= (3/4)**dy
        return True

    def update(self, dt):
        if self.anim_zoom_center is not None:
            r = self.anim_extent_x / self.extent_x
            self._zoom(r**(dt*16), self.anim_zoom_center)
            if abs(r-1.0) < 1e-3:
                self.anim_zoom_center = None

    def _zoom(self, scale_step, zoom_center):
        self.center = (self.center - zoom_center)*scale_step + zoom_center
        self.extent_x *= scale_step
        
    def scr2wld(self, x, y):
        w, h = self.size
        pe = self.get_pixel_extent()
        return self.center + pe * ((x, y)-V(w, h)/2)
    def get_pixel_extent(self):
        w, h = self.size
        dx = self.extent_x / w
        return V(dx, dx)

    def __enter__(self):
        w, h = self.size
        cx, cy = self.center
        dx, dy = 0.5*self.get_pixel_extent()*(w, h)

        gl.glViewport(0, 0, w, h)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.gluOrtho2D(cx-dx, cx+dx, cy-dy, cy+dy)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
    def __exit__(self, *args):
        pass

class OverlayView(object):
    def __init__(self, window):
        self.window = window
    def __enter__(self):
        width, height = self.window.get_size()
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, width, 0, height, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
    def __exit__(self, *arg):
        pass

name2gltype = dict(
    uint8   = gl.GL_UNSIGNED_BYTE,
    float32 = gl.GL_FLOAT)
def arrayToGLType(a):
    return name2gltype[a.dtype.name]

ascont = np.ascontiguousarray

class glstate:
    ClientState = set([
      gl.GL_VERTEX_ARRAY, 
      gl.GL_TEXTURE_COORD_ARRAY, 
      gl.GL_COLOR_ARRAY, 
      gl.GL_NORMAL_ARRAY, 
      gl.GL_SECONDARY_COLOR_ARRAY, 
      gl.GL_EDGE_FLAG_ARRAY])

    def __init__(self, *state):
        self.state = state
    def __enter__(self):
        for idx in self.state:
            texuint = 0
            if hasattr(idx, "__len__"):
                texunit = idx[1]
                idx = idx[0]
            if idx in self.ClientState:
                if idx == gl.GL_TEXTURE_COORD_ARRAY:
                    gl.glClientActiveTexture(GL_TEXTURE0 + texunit)
                gl.glEnableClientState(idx)
            else:
                gl.glEnable(idx)
    def __exit__(self, *args):
        for idx in self.state:
            texuint = 0
            if hasattr(idx, "__len__"):
                texunit = idx[1]
                idx = idx[0]
            if idx in self.ClientState:
                if idx == gl.GL_TEXTURE_COORD_ARRAY:
                    gl.glClientActiveTexture(GL_TEXTURE0 + texunit)
                gl.glDisableClientState(idx)
            else:
                gl.glDisable(idx)


def draw_arrays(primitive, verts=None, colors=None, tc0=None, tc1=None, tc2=None, tc3=None):
    states = []
    if verts is not None:
        verts = ascont(verts)
        assert verts.ndim >= 2 and verts.shape[-1] <= 4
        gl.glVertexPointer( verts.shape[-1], arrayToGLType(verts), verts.strides[-2], verts.ctypes.data)
        states.append( gl.GL_VERTEX_ARRAY )
    if colors is not None:
        colors = ascont(colors)
        gl.glColorPointer( colors.shape[-1], arrayToGLType(colors), colors.strides[-2], colors.ctypes.data)
        states.append( gl.GL_COLOR_ARRAY )
    protect = []
    for i, tc in enumerate([tc0, tc1, tc2, tc3]):
        if tc is not None:
            tc = ascont(tc)
            protect.append(tc)
            gl.glClientActiveTexture(GL_TEXTURE0 + i) 
            gl.glTexCoordPointer(tc.shape[-1], arrayToGLType(tc), tc.strides[-2], tc.ctypes.data)
            states.append( (GL_TEXTURE_COORD_ARRAY, i) )
    with glstate(*states):
        gl.glDrawArrays( primitive, 0, np.prod(verts.shape[:-1]) )
