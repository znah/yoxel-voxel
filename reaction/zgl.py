from __future__ import with_statement
import sys
import wx
from wx import glcanvas
from OpenGL.GL import *
from OpenGL.extensions import alternate
from OpenGL.GLU import *
from OpenGL.GL.EXT.framebuffer_object import *
from OpenGL.GL.ARB.framebuffer_object import *
from OpenGL.GL.EXT.texture_integer import *
from OpenGL.GL.EXT.texture_array import *
from numpy import *
from time import clock
import PIL.Image
from enthought.traits.api import *
from enthought.traits.ui.api import *
from StringIO import StringIO
import re

from ctypes import cdll, c_int, c_uint, c_float, c_char_p, c_long


############### framebuffer_object wrappers ##############
glGenFramebuffers = wrapper.wrapper(glGenFramebuffers).setOutput(
                'framebuffers', 
                lambda x: (x,), 
                'n')
                
glGenRenderbuffers = wrapper.wrapper(glGenRenderbuffers).setOutput(
                'renderbuffers', 
                lambda x: (x,), 
                'n')
#############################################################

def FixFramebufferAPI():
    if not glInitFramebufferObjectARB():
        import OpenGL.GL.EXT.framebuffer_object as ext
        names = [s for s in dir(ext) if s[:2] == 'gl' and s[-3:] == 'EXT']
        for name in names:
            globals()[ name[:-3] ] = getattr(ext, name)

############### wglSwapIntervalEXT workaround ###############
import OpenGL.platform
wglSwapIntervalEXT = OpenGL.platform.createExtensionFunction( 
  'wglSwapIntervalEXT', dll=OpenGL.platform.GL,
  extension='WGL_EXT_swap_control',
  resultType=c_long, 
  argTypes=[c_int],
  doc='wglSwapIntervalEXT( c_int(None) ) -> BOOL', 
  argNames=['None'],
)
#############################################################

cg = cdll.LoadLibrary("cg.dll")
cggl = cdll.LoadLibrary("cggl.dll")

cgGetError = cg.cgGetError
cgGetErrorString = cg.cgGetErrorString
cgGetErrorString.restype = c_char_p
cgGetLastListing = cg.cgGetLastListing
cgGetLastListing.restype = c_char_p

cgGLEnableProfile  = cggl.cgGLEnableProfile
cgGLDisableProfile = cggl.cgGLDisableProfile
cgGLBindProgram    = cggl.cgGLBindProgram
cgGLUnbindProgram  = cggl.cgGLUnbindProgram

cgGetNamedParameter = cg.cgGetNamedParameter
cgGetNamedParameter.argtypes = [c_int, c_char_p]

cgGetParameterType = cg.cgGetParameterType

cgGLSetParameter1f = cggl.cgGLSetParameter1f
cgGLSetParameter1f.argtypes = [c_int, c_float]

cgGLSetParameter2f = cggl.cgGLSetParameter2f
cgGLSetParameter2f.argtypes = [c_int, c_float, c_float]

cgGLSetParameter3f = cggl.cgGLSetParameter3f
cgGLSetParameter3f.argtypes = [c_int, c_float, c_float, c_float]

cgGLSetParameter4f = cggl.cgGLSetParameter4f
cgGLSetParameter4f.argtypes = [c_int, c_float, c_float, c_float, c_float]

cgSetParameter1i = cg.cgSetParameter1i
cgSetParameter1i.argtypes = [c_int, c_int]

cgGLSetTextureParameter = cggl.cgGLSetTextureParameter
cgGLSetTextureParameter.argtypes = [c_int, c_uint]


cgProfiles = {"fp30"  : 6149, 
              "vp30"  : 6148,
              "fp40"  : 6151, 
              "vp40"  : 7001, 
              "gp4fp" : 7010,
              "gp4vp" : 7011,
              "gp4gp" : 7012}

cgParamSetters = {
  cg.cgGetType("float")     : lambda p, v : cgGLSetParameter1f(p, v),
  cg.cgGetType("int")       : lambda p, v : cgSetParameter1i(p, v),
  cg.cgGetType("float2")    : lambda p, v : cgGLSetParameter2f(p, v[0], v[1]),
  cg.cgGetType("float3")    : lambda p, v : cgGLSetParameter3f(p, v[0], v[1], v[2]),
  cg.cgGetType("float4")    : lambda p, v : cgGLSetParameter4f(p, v[0], v[1], v[2], v[3]),
  cg.cgGetType("sampler1D") : lambda p, v : cgGLSetTextureParameter(p, v),
  cg.cgGetType("sampler2D") : lambda p, v : cgGLSetTextureParameter(p, v),
  cg.cgGetType("sampler3D") : lambda p, v : cgGLSetTextureParameter(p, v),
  cg.cgGetType("sampler2DARRAY") : lambda p, v : cgGLSetTextureParameter(p, v)
}

def arrayToGLType(a):
    if a.dtype.char == 'l':
        return GL_INT
    elif a.dtype.char == 'L':
        return GL_UNSIGNED_INT
    return arrays.ArrayDatatype.arrayToGLType(a)

def InitCG():
    global cgContext
    cgContext = cg.cgCreateContext()
    cggl.cgGLSetManageTextureParameters(cgContext, 1)
    
    def InitProfile(name):
        if cggl.cgGLIsProfileSupported(cgProfiles[name]) != 0:
            cggl.cgGLSetOptimalOptions(cgProfiles[name])
        else:
            print "profile", name, "not supported"
    for profile in cgProfiles:
        InitProfile(profile)

def checkCGerror():
    err = cgGetError()
    if err == 0:
        return
    msg = cgGetErrorString(err)
    listing = cgGetLastListing(cgContext)
    print listing
    raise Exception(msg, listing)

def V(*args):
    if len(args) == 1:
        return array(args[0], float32)
    else:
        return array(args, float32)

def YX(*args):
    if len(args) == 1:
        return array((args[0][1], args[0][0]), float32)
    elif len(args) == 2:
        return array((args[1], args[0]), float32)
    else:
        return array([args[1], args[0]] + list(args[2:]), float32)
                            
class CGShader:
    def __init__(self, profileName, code = None, fileName = None, entry = "main"):
        CG_SOURCE = 4112
        self._profile = cgProfiles[profileName]
        if code is not None:
            self._prog = cg.cgCreateProgram(cgContext, CG_SOURCE, code, self._profile, entry, None)
        else:
            self._prog = cg.cgCreateProgramFromFile(cgContext, CG_SOURCE, fileName, self._profile, entry, None)
        checkCGerror()
        cggl.cgGLLoadProgram(self._prog)
        checkCGerror()

    def __del__(self):
        cg.cgDestroyProgram(self._prog)

    def __enter__(self):
        cgGLEnableProfile(self._profile)
        cgGLBindProgram(self._prog)

    def __exit__(self, *args):
        cgGLUnbindProgram(self._profile)
        cgGLDisableProfile(self._profile)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            self.__dict__[name] = value
            return
        param = cgGetNamedParameter(self._prog, name)
        if param == 0:
            raise Exception("unknown shader param", name)
        type_ = cgGetParameterType(param)
        cgParamSetters[type_](param, value)
        self.__dict__[name] = value

    def __call__(self, **args):
        for name in args:
            setattr(self, name, args[name])
        return self

class Texture(object):
    Target = None  # to be set by subclass

    ChNum2Format = {1:GL_LUMINANCE, 3:GL_RGB, 4:GL_RGBA}
    
    Nearest = [(GL_TEXTURE_MIN_FILTER, GL_NEAREST), (GL_TEXTURE_MAG_FILTER, GL_NEAREST)]
    Linear  = [(GL_TEXTURE_MIN_FILTER, GL_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR)]
    MipmapLinear  = [(GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR)]
    Repeat  = [(GL_TEXTURE_WRAP_S, GL_REPEAT), (GL_TEXTURE_WRAP_T, GL_REPEAT), (GL_TEXTURE_WRAP_R, GL_REPEAT)]
    Clamp  = [(GL_TEXTURE_WRAP_S, GL_CLAMP), (GL_TEXTURE_WRAP_T, GL_CLAMP), (GL_TEXTURE_WRAP_R, GL_CLAMP)]
    ClampToEdge  = [(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)]

    def __init__(self):
        self._as_parameter_ = glGenTextures(1)

    def filterNearest(self):
        self.setParams(*self.Nearest)
    def filterLinear(self):
        self.setParams(*self.Linear)
    def filterLinearMipmap(self):
        self.setParams(*self.MipmapLinear)

    def setParams(self, *args):
        with self:
            for pname, val in args:
                glTexParameteri(self.Target, pname, val)

    def genMipmaps(self):
        with self:
            glGenerateMipmap(self.Target)

    def aniso(self, n):
        self.setParams( (GL_TEXTURE_MAX_ANISOTROPY_EXT, n))
    
    def __enter__(self):
        glBindTexture(self.Target, self)

    def __exit__(self, *args):
        glBindTexture(self.Target, 0)
    
    def __del__(self):
        self.free()
    def free(self):
        glDeleteTextures([self._as_parameter_])



class Texture1D(Texture):
    Target = GL_TEXTURE_1D

    def __init__(self, img = None, size = None, format = GL_RGBA8, srcFormat = None, srcType = None):
        Texture.__init__(self)
        self.setParams( *(self.Nearest + self.Repeat) )
        if img != None:
            with self:
                img = atleast_2d(ascontiguousarray(img))
                if srcFormat is None:
                    srcFormat = self.ChNum2Format[img.shape[1]]
                if srcType is None:
                    srcType = arrayToGLType(img)
                glPixelStorei(GL_PACK_ALIGNMENT, 1);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                glTexImage1D(self.Target, 0, format, img.shape[0], 0, srcFormat, srcType, img)
                self.size = img.shape[0]
            return
        elif size != None:
            with self:
                if srcFormat is None:
                    srcFormat = GL_RGBA
                if srcType is None:
                    srcType = GL_FLOAT
                glTexImage1D(self.Target, 0, format, size, 0, srcFormat, srcType, None)
                self.size = size

class Texture2D(Texture):
    Target = GL_TEXTURE_2D

    def __init__(self, img = None, size = None, format = GL_RGBA8, srcFormat = GL_RGBA, srcType = GL_FLOAT):
        Texture.__init__(self)
        self.setParams( *(self.Nearest + self.Repeat) )
        if img != None:
            with self:
                img = atleast_3d(ascontiguousarray(img))
                srcFormat = self.ChNum2Format[img.shape[2]]
                srcType = arrayToGLType(img)
                glPixelStorei(GL_PACK_ALIGNMENT, 1);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                glTexImage2D(self.Target, 0, format, img.shape[1], img.shape[0], 0, srcFormat, srcType, img)
                self.size = YX(img.shape[:2])
            return
        elif size != None:
            with self:
                glTexImage2D(self.Target, 0, format, size[0], size[1], 0, srcFormat, srcType, None)
                self.size = size

class Texture3D(Texture):
    Target = GL_TEXTURE_3D

    def __init__(self, img = None, size = None, format = GL_RGBA8, srcFormat = GL_RGBA, srcType = GL_FLOAT):
        Texture.__init__(self)
        self.setParams( *(self.Nearest + self.Repeat) )
        if img != None:
            with self:
                img = atleast_3d(ascontiguousarray(img))
                if img.ndim == 3:
                    ch = 1
                else:
                    ch = img.shape[3]
                srcFormat = self.ChNum2Format[ch]
                srcType = arrayToGLType(img)
                glPixelStorei(GL_PACK_ALIGNMENT, 1);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                glTexImage3D(self.Target, 0, format, img.shape[2], img.shape[1], img.shape[0], 0, srcFormat, srcType, img)
                self.size = V(img.shape[::-1])
            return
        elif size != None:
            with self:
                glTexImage3D(self.Target, 0, format, size[0], size[1], size[2], 0, srcFormat, srcType, None)
                self.size = size

class TextureArray(Texture3D):
    Target = GL_TEXTURE_2D_ARRAY_EXT
    def __init__(self, *args, **kargs):
        Texture3D.__init__(self, *args, **kargs)

class BufferObject:
    def __init__(self, data = None, use = GL_STATIC_DRAW):
        self.handle = int(glGenBuffers(1))
        self._as_parameter_ = self.handle

        class Binder:
            def __init__(self, parent, target):
                self.parent = parent
                self.target = target
            def __enter__(self):
                glBindBuffer(self.target, self.parent)
            def __exit__(self, *args):
                glBindBuffer(self.target, 0)
        self.pixelPack    = Binder(self, GL_PIXEL_PACK_BUFFER)
        self.pixelUnpack  = Binder(self, GL_PIXEL_UNPACK_BUFFER)
        self.array        = Binder(self, GL_ARRAY_BUFFER)
        self.elementArray = Binder(self, GL_ELEMENT_ARRAY_BUFFER)

        if data is not None:
            with self.array:
                glBufferData(GL_ARRAY_BUFFER, data, use)
                
    def __del__(self):
        glDeleteBuffers([self._as_parameter_])


class Framebuffer:
    def __init__(self):
        self._as_parameter_ = glGenFramebuffers(1)
    def __del__(self):
        glDeleteFramebuffers(1, [self._as_parameter_])
    def __enter__(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self)
    def __exit__(self, *args):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

class Renderbuffer:
    def __init__(self):
        self._as_parameter_ = glGenRenderbuffers(1)
    def __del__(self):
        glDeleteRenderbuffers(1, [self._as_parameter_])
    def __enter__(self):
        glBindRenderbuffer(GL_RENDERBUFFER, self)
    def __exit__(self, *args):
        glBindRenderbuffer(GL_RENDERBUFFER, 0)


class RenderTexture:
    def __init__(self, depth = False, **args):
        self.fbo = Framebuffer()
        self.tex = Texture2D(**args)
        with self.fbo:
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.tex, 0)
            if depth:
                self.depthTex = Texture2D(size=self.size(), format=GL_DEPTH_COMPONENT24, srcFormat=GL_DEPTH_COMPONENT)
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depthTex, 0)

    def size(self):
        return self.tex.size

    def __enter__(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.size()[0], self.size()[1])

    def __exit__(self, *args):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def texparams(self, *args):
        self.tex.setParams(*args)


class ctx:
    def __init__(self, *items):
        self.items = items
    def __enter__(self):
        self.exitlist = []
        try:
            for item in self.items:
                item.__enter__()
                self.exitlist.append(item)
        except:
            for item in reversed(self.exitlist):
                item.__exit__(*sys.exc_info())
            raise
    def __exit__(self, *args):
        for item in reversed(self.exitlist):
            item.__exit__(*args)
        del self.exitlist

class Ortho:
    def __init__(self, rect = (0, 0, 1, 1)):
        self.rect = rect
    def __enter__(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        (x1, y1, x2, y2) = self.rect
        gluOrtho2D(x1, x2, y1, y2)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
    def __exit__(self, *args):
        pass

ortho = Ortho()

class PingPong:
    def __init__(self, **args):
        (self.src, self.dst) = [RenderTexture(**args) for i in (1, 2)]
    def size(self):
        return self.src.size()
    def flip(self):
        (self.src, self.dst) = (self.dst, self.src)
    def texparams(self, *args):
        self.src.texparams(*args)
        self.dst.texparams(*args)

def drawQuad(rect = (0, 0, 1, 1)):
    glBegin(GL_QUADS)
    def vert(x, y):
        glTexCoord2f(x, y)
        glVertex2f(x, y)
    (x1, y1, x2, y2) = rect
    vert(x1, y1)
    vert(x2, y1)
    vert(x2, y2)
    vert(x1, y2)
    glEnd()

def drawVerts(primitive, pos, texCoord = None):
    glBegin(primitive)
    for i in xrange(len(pos)):
        if texCoord is not None: 
            glTexCoord(*texCoord[i])
        glVertex(*pos[i])
    glEnd()
    
class Viewport:
    def __init__(self, x = 0, y = 0, width = 1, height = 1):
        self.origin = (x, y)
        self.size = (width, height)
    def __enter__(self):
        glViewport(self.origin[0], self.origin[1], self.size[0], self.size[1])
    def __exit__(self, *args):
        pass
    def aspect(self):
        return float(self.size[0]) / self.size[1]

class WXAdapter:
    wx2glut = {wx.MOUSE_BTN_LEFT : 0, wx.MOUSE_BTN_MIDDLE: 1, wx.MOUSE_BTN_RIGHT: 2}

    def OnMouse(self, evt):
        x, y = evt.Position
        wheel = evt.GetWheelRotation()
        if evt.IsButton():
            up = evt.ButtonUp()            
            btn = self.wx2glut[evt.GetButton()]
            self.mouseButton(btn, up, x, y)
        elif wheel > 0:
            self.mouseButton(3, 0, x, y)
        elif wheel < 0:
            self.mouseButton(4, 0, x, y)
        else:
            self.mouseMove(x, y)


class FlyCamera(WXAdapter):
    def __init__(self):
        self.eye = (0.0, 0.0, 0.0)
        self.course = 0
        self.pitch = 0
        self.fovy = 40
        self.vp = Viewport()
        self.zNear = 1.0
        self.zFar  = 1000.0
        
        self.mButtons = zeros((3,), bool) # left, middle, right
        self.mPos = (0, 0)
        
        self.vel = [0, 0, 0]
        self.sensitivity = 0.3
        self.speed = 1.0

        self.boost = False

        self.with_vp = ctx(self.vp, self)

    def __setattr__(self, name, val):
        if name == "eye":
            val = V(*val)
        self.__dict__[name] = val

    def resize(self, x, y):
        self.vp.size = (x, max(y, 1))

    key2vel = {'w': (0, 1), 's': (0, -1), 'd': (1, 1), 'a': (1, -1), 'q': (2, 1), 'z': (2, -1) }

    def keyDown(self, key, x, y):
        k = key.lower()
        if self.key2vel.has_key(k):
            a, d = self.key2vel[k]
            self.vel[a] = d

    def keyUp(self, key, x, y):
        k = key.lower()
        if self.key2vel.has_key(k):
            a, d = self.key2vel[k]
            self.vel[a] = 0

    def OnKeyDown(self, evt):
        key = evt.GetKeyCode()
        if key < 256:
            k = chr(key).lower()
            if self.key2vel.has_key(k):
                a, d = self.key2vel[k]
                self.vel[a] = d
        elif key == wx.WXK_SHIFT:
            self.boost = True

    def OnKeyUp(self, evt):
        key = evt.GetKeyCode()
        if key < 256:
            k = chr(key).lower()
            if self.key2vel.has_key(k):
                a, d = self.key2vel[k]
                self.vel[a] = 0
        elif key == wx.WXK_SHIFT:
            self.boost = False
            
    def mouseMove(self, x, y):
        dx = x - self.mPos[0]
        dy = y - self.mPos[1]
        if self.mButtons[0]:
            self.course -= dx * self.sensitivity
            self.pitch = clip(self.pitch - dy * self.sensitivity, -89.9, 89.9)
        self.mPos = (x, y)

    def mouseButton(self, btn, up, x, y):
        if btn < 3:
            self.mButtons[btn] = not up
        if btn == 3 and self.fovy > 0.5 :
            self.sensitivity /= 1.1
            self.fovy /= 1.1
        if btn == 4 and self.fovy < 90:
            self.sensitivity *= 1.1
            self.fovy *= 1.1
        self.mPos = (x, y)

    def __enter__(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fovy, self.vp.aspect(), self.zNear, self.zFar)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        args = []
        args.extend(self.eye)
        args.extend(self.eye + self.forwardVec())
        args.extend([0, 0, 1])  # up
        gluLookAt(*args)
    def __exit__(self, *args):
        pass
        
    def forwardVec(self):
        (c, p) = radians((self.course, self.pitch))
        s = cos(p)
        return array( [s*cos(c), s*sin(c), sin(p)] )
    def rightVec(self):
        fwd = self.forwardVec()
        up = array([0, 0, 1])
        right = cross(fwd, up)
        right /= linalg.norm(right)
        return right
    def viewUpVec(self):
        fwd   = self.forwardVec()
        right = self.rightVec()
        return cross(right, fwd)
    
    def update(self, dt):
        v = self.vel[0] * self.forwardVec() + self.vel[1] * self.rightVec() + self.vel[2] * self.viewUpVec()
        if self.boost != 0:
            v *= 10
        self.eye += v*self.speed*dt


class OrthoCamera(WXAdapter):
    def __init__(self):
        self.vp = Viewport()
        self.rect = (0.0, 0.0, 1.0, 1.0)
        self.unsized = True

        self.mButtons = zeros((3,), bool)  # left, middle, right
        self.mPos = (0, 0)
        self.mPosWld = (0, 0)
        self.with_vp = ctx(self.vp, self)
   
    def extent(self):
        (x1, y1, x2, y2) = self.rect
        return V(x2-x1, y2-y1)

    def resize(self, x, y):
        oldSize = self.vp.size
        self.vp.size = V(max(x, 1), max(y, 1))
        if self.unsized:
            (x1, y1, x2, y2) = self.rect
            vr = self.vp.aspect()
            rr = (y2-y1) / (x2-x1)
            if vr > rr:
                h = 0.5*(y2-y1)*vr
                c = 0.5*(x1+x2)
                self.rect = (c-h, y1, c+h, y2)
            else:
                h = 0.5*(x2-x1)/vr
                c = 0.5*(y1+y2)
                self.rect = (x1, c-h, x2, c+h)
            self.unsized = False
            return
            
        r = self.vp.size / oldSize
        (x1, y1, x2, y2) = self.rect
        p1 = V(x1, y1)
        p2 = V(x2, y2)
        c  = 0.5 * (p1 + p2)
        sz = 0.5*(p2-p1)*r
        p1 = c - sz
        p2 = c + sz
        self.rect = (p1[0], p1[1], p2[0], p2[1])

    def mouseMove(self, x, y):
        dx = x - self.mPos[0]
        dy = y - self.mPos[1]
        if self.mButtons[0]:
            (sx, sy) = V(-dx, dy) / self.vp.size * self.extent()
            (x1, y1, x2, y2) = self.rect
            self.rect = (x1+sx, y1+sy, x2+sx, y2+sy)
        self.mPos = (x, y)
        self.mPosWld = self.scr2wld(x, y)

    def scr2wld(self, x, y):
        (x1, y1, x2, y2) = self.rect
        (sx, sy) = V(x, y) / self.vp.size
        return V(x1 + (x2-x1)*sx, y1 + (y2-y1)*(1.0-sy))

    def lo(self):
        return V(self.rect[0:2])
    def hi(self):
        return V(self.rect[2:4])

    def scale(self, coef, center):
        lo = center + (self.lo() - center)*coef
        hi = center + (self.hi() - center)*coef
        self.rect = (lo[0], lo[1], hi[0], hi[1])
                                              
    def mouseButton(self, btn, up, x, y):
        self.mPos = (x, y)
        self.mPosWld = self.scr2wld(x, y)
        if btn < 3:
            self.mButtons[btn] = not up
        if btn == 3:
            self.scale(1.0/1.1, self.mPosWld)
        if btn == 4:
            self.scale(1.1, self.mPosWld)

    def __enter__(self):
        self.proj = Ortho(self.rect)
        self.proj.__enter__()
    def __exit__(self, *args):
        self.proj.__exit__()

    def update(self, dt):
        pass


def safe_call(obj, method, *l, **d):
    if hasattr(obj, method):
        getattr(obj, method)(*l, **d)

class ZglAppWX(HasTraits):
    _ = Python(editable = False)

    def __init__(self, title = "ZglAppWX", size = (800, 600), viewControl = None, vsync = 0, zglpath = '.'):
        HasTraits.__init__(self)
        self.app = wx.PySimpleApp()
        self.frame = frame = wx.Frame(None, wx.ID_ANY, title)
        self.canvas = canvas = glcanvas.GLCanvas(frame, -1)
        self.initSize = size
        self.viewControl = viewControl
        self.viewSize = size

        canvas.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
        canvas.Bind(wx.EVT_PAINT, self.OnPaint)

        canvas.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        canvas.Bind(wx.EVT_KEY_UP, self.OnKeyUp)
        canvas.Bind(wx.EVT_MOUSE_EVENTS, self.OnMouse)

        canvas.Bind(wx.EVT_IDLE, self.OnIdle)
        
        frame.Bind(wx.EVT_CLOSE, self.OnClose)
       
        frame.Show(True)
        canvas.SetCurrent()
        InitCG()

        FixFramebufferAPI()
        
        wglSwapIntervalEXT(vsync)

        self._textFrag = CGShader('fp40', '''
          uniform sampler2D fontTex;
          uniform float4 color;
          uniform float2 dp;
          float4 main(float2 pos : TEXCOORD0) : COLOR
          {
            float c = tex2D(fontTex, pos);
            return color * c;
          }
        ''')
        font = asarray(PIL.Image.open(zglpath + "/img/font.png").convert('L')).copy()
        self._textFrag.fontTex = fontTex = Texture2D(font)
        self._textFrag.dp = 1.0 / fontTex.size

        def debugNone(reset):
            global g_profileEnable
            if reset:
                g_profileEnable = False
        def debugProfile(reset):
            global g_profileEnable, _profileNodes
            if reset:
                g_profileEnable = True
                _profileNodes = {}
            else:
                self.drawText((50, 50), dumpProfile())
        import itertools
        self.debugFuncIter = itertools.cycle( [debugNone, debugProfile] )
        self.debugFunc = self.debugFuncIter.next()
        self.debugFunc(True)

        self.initKeyTable()

    def initKeyTable(self):
        codes =  [(getattr(wx, s), s[4:]) for s in dir(wx) if s[:4] == 'WXK_']
        codes += [ (code, chr(code)) for code in xrange( ord('A'), ord('Z')+1 )]
        codes += [ (code, chr(code)) for code in xrange( ord('0'), ord('9')+1 )]
        self.key2name = dict(codes)

    def run(self):
        self.canvas.Bind(wx.EVT_SIZE, self.OnSize)
        self.frame.SetClientSize(self.initSize)
        self.app.SetTopWindow(self.frame)
        
        self.time = clock()
        self.dt = 0
        self.app.MainLoop()

    def OnClose(self, event):
        self.app.Exit()

    def OnIdle(self, event):
        t = clock()
        self.dt = t - self.time
        self.time = t
        self.viewControl.update(self.dt)
        safe_call(self, 'update', t, self.dt)
        self.canvas.Refresh(False)

    def OnEraseBackground(self, event):
        pass # Do nothing, to avoid flashing on MSW.

    
    def OnPaint(self, event):
        dc = wx.PaintDC(self.canvas)
        safe_call(self, "display")
        self.debugFunc(False)
        self.canvas.SwapBuffers()

    def display(self):
        glClearColor(0, 0.5, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def key_ESCAPE(self):
        self.frame.Close(True)
    
    def key_F1(self):
        self.debugFunc = self.debugFuncIter.next()
        self.debugFunc(True)

    def key_F2(self):
        self.edit_traits() 
    
    def OnKeyDown(self, event):
        code = event.GetKeyCode()
        if code in self.key2name:
            safe_call(self, 'key_' + self.key2name[code])
        safe_call(self.viewControl, 'OnKeyDown', event)

    def OnKeyUp(self, event):
        safe_call(self.viewControl, 'OnKeyUp', event)

    def OnMouse(self, evt):
        safe_call(self.viewControl, 'OnMouse', evt)

    def OnSize(self, event):
        x, y = self.canvas.GetClientSize()
        safe_call(self.viewControl, 'resize', x, y)
        self.viewSize = self.viewControl.vp.size
        safe_call(self, 'resize', x, y)
        self.canvas.Refresh(False)

    def drawText(self, pos, s, color = (0.5, 1, 0.5, 1)):
        sx, sy = self.viewControl.vp.size
        rect = (0, sy, sx, 0)
        with ctx( self._textFrag(color = color), glstate(GL_BLEND), Ortho(rect) ):
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glBlendEquation(GL_FUNC_ADD);

            w, h = 8, 16
            x, y = pos
            v = array([[0, 0], [w, 0], [w, h], [0, h]], float32)
            ss = s.splitlines()
            total = sum(map(len, ss))
            max_n = max(map(len, ss))
            px = arange(max_n)*w + x
            pos = tile(v, (total, 1, 1))
            tc  = tile(v, (total, 1, 1))
            ofs = 0
            for s in ss:
                ch = fromstring(s, uint8)
                n = len(s)
                cx = (ch % 16) * w
                cy = (ch / 16) * h

                linePos = pos[ofs:][:n]
                linePos[...,0] += px[:n,newaxis]
                linePos[...,1] += y
                lineTC = tc[ofs:][:n]
                lineTC[..., 0] += cx[:,newaxis]
                lineTC[..., 1] += cy[:,newaxis]

                y += h
                ofs += n
            tc /= (128, 256)
            drawArrays(GL_QUADS, verts = pos, tc0 = tc)    





class vattr:
    def __init__(self, *idxs):
        self.idxs = idxs
    def __enter__(self):
        [glEnableVertexAttribArray(idx)  for idx in self.idxs]
    def __exit__(self, *args):
        [glDisableVertexAttribArray(idx)  for idx in self.idxs]

class glstate:
    ClientState = set([
      GL_VERTEX_ARRAY, 
      GL_TEXTURE_COORD_ARRAY, 
      GL_COLOR_ARRAY, 
      GL_NORMAL_ARRAY, 
      GL_SECONDARY_COLOR_ARRAY, 
      GL_EDGE_FLAG_ARRAY])

    def __init__(self, *state):
        self.state = state
    def __enter__(self):
        for idx in self.state:
            texuint = 0
            if hasattr(idx, "__len__"):
                texunit = idx[1]
                idx = idx[0]
            if idx in self.ClientState:
                if idx == GL_TEXTURE_COORD_ARRAY:
                    glClientActiveTexture(GL_TEXTURE0 + texunit)
                glEnableClientState(idx)
            else:
                glEnable(idx)
    def __exit__(self, *args):
        for idx in self.state:
            texuint = 0
            if hasattr(idx, "__len__"):
                texunit = idx[1]
                idx = idx[0]
            if idx in self.ClientState:
                if idx == GL_TEXTURE_COORD_ARRAY:
                    glClientActiveTexture(GL_TEXTURE0 + texunit)
                glDisableClientState(idx)
            else:
                glDisable(idx)

def loadTex(fn):
    tex = Texture2D(PIL.Image.open(fn))
    tex.filterLinearMipmap()
    tex.genMipmaps()
    tex.aniso(8)
    return tex



_ObejctPool = {}

def drawGrid(w, h = None, normalized = True):
    if h is None:
        h = w
    objName = "drawGrid_%dx%d" % (w, h)
    if objName not in _ObejctPool:
        verts = zeros((h, w, 2), float32)
        verts[...,1], verts[...,0] = indices((h, w))
        if normalized:
            verts /= (w, h)
    
        idxgrid = arange(h*w).reshape(h, w)
        idxs= zeros((h-1, w-1, 4), uint32)
        idxs[...,0] = idxgrid[ :-1, :-1 ]
        idxs[...,1] = idxgrid[ :-1,1:   ]  
        idxs[...,2] = idxgrid[1:  ,1:   ]
        idxs[...,3] = idxgrid[1:  , :-1 ]
        idxs = idxs.flatten()

        vertBuf = BufferObject(verts)
        idxBuf = BufferObject(idxs)
        _ObejctPool[objName] = (vertBuf, idxBuf, len(idxs))

    (vertBuf, idxBuf, idxNum) = _ObejctPool[objName]
    with vertBuf.array:
        glVertexAttribPointer(0, 2, GL_FLOAT, False, 0, None)
    with ctx(idxBuf.elementArray, vattr(0)):
        glDrawElements(GL_QUADS, idxNum, GL_UNSIGNED_INT, None)
   

def clearGLBuffers(color = (0, 0, 0, 0), colorBit = True, depthBit = True):
    mask = 0
    if colorBit:
       glClearColor(*color)
       mask |= GL_COLOR_BUFFER_BIT
    if depthBit:
       mask |= GL_DEPTH_BUFFER_BIT
    glClear(mask)
        
def load_obj(fn):
    ss = file(fn).readlines()
    vs = [s[1:] for s in ss if s[0] == 'v']
    fs = [s[1:] for s in ss if s[0] == 'f']
    verts = loadtxt( StringIO("".join(vs)), float32 )
    faces = loadtxt( StringIO("".join(fs)), int32 ) - 1
    return (verts, faces)

def save_obj(fn, verts, faces):
    f = file(fn, 'w')
    f.write("# verts: %d\n# faces: %d\n\n" % (len(verts), len(faces)))
    for v in verts:
        f.write("v %f %f %f\n" % tuple(v))
    for face in faces:
        f.write("f %d %d %d\n" % tuple(face+1))
    f.close()

def create_box():
    verts = indices((2, 2, 2), float32).T.reshape(-1,3)
    idxs = [ 0, 2, 3, 1,
             0, 1, 5, 4, 
             4, 5, 7, 6,
             1, 3, 7, 5,
             0, 4, 6, 2,
             2, 6, 7, 3]
    quad_idxs = array(idxs, uint32).reshape(-1,4)
    trg_idxs = quad_idxs[:,(0, 1, 2, 0, 2, 3)].reshape((-1,3))  # quads -> triangles
    return verts, trg_idxs, quad_idxs

    
def drawArrays(primitive, verts = None, indices = None, tc0 = None, normals = None):
    states = []
    if verts is not None:
        glVertexPointer( verts.shape[-1], arrayToGLType(verts), verts.strides[-2], verts)
        states.append( GL_VERTEX_ARRAY )
    if normals is not None:
        glNormalPointer( arrayToGLType(verts), verts.strides[-2], normals)
        states.append( GL_NORMAL_ARRAY )
    if tc0 is not None:
       glClientActiveTexture(GL_TEXTURE0) 
       glTexCoordPointer(tc0.shape[-1], arrayToGLType(tc0), tc0.strides[-2], tc0)
       states.append( (GL_TEXTURE_COORD_ARRAY, 0) )
    with glstate(*states):
        if indices is not None:
            # TODO: index types other that uint32
            glDrawElements(primitive, indices.size, GL_UNSIGNED_INT, indices)
        else:
            glDrawArrays( primitive, 0, prod(verts.shape[:-1]) )



_profileNodes = {}
_profileCurNodeName = ""
g_profileEnable = False           

class profile:
    def __init__(self, name, log = None):
        self.nodeName = name
        self.log = log
            
    def __enter__(self):
        if not g_profileEnable:
            return
        global _profileCurNodeName
        self.prevNodeName = _profileCurNodeName
        if _profileCurNodeName == "":
            self.fullName = self.nodeName
        else:
            self.fullName = _profileCurNodeName + "." + self.nodeName
        _profileCurNodeName = self.fullName
        self.startTime = clock()
    def __exit__(self, *args):
        if not g_profileEnable:
            return
        global _profileCurNodeName
        dt = (clock() - self.startTime) * 1000.0
        if self.fullName in _profileNodes:
            node_data = _profileNodes.get( self.fullName )
            node_data['ncalls'] += 1
            node_data['total']  += dt
            node_data['max']    = max(dt, node_data['max'])
            relaxCoef = 0.1 #min(0.5, dt / 50.0)
            node_data['avg']    = (1.0 - relaxCoef) * node_data['avg'] + relaxCoef * dt
        else:
            node_data = dict(ncalls=1, total = dt, max = dt, avg = dt)
        if self.log is not None:
            log = node_data.get('log', [])
            if not iterable(self.log):
                self.log = [self.log]

            log.append([dt] + list(self.log))
            node_data['log'] = log
        _profileNodes[self.fullName] = node_data
        _profileCurNodeName = self.prevNodeName 

class glprofile(profile):
    def __init__(self, name, log = None):
        profile.__init__(self, name + '_gl', log)
    def __enter__(self):
        if not g_profileEnable:
            return
        glFinish()
        profile.__enter__(self)
    def __exit__(self, *argd):
        if not g_profileEnable:
            return
        glFinish()
        profile.__exit__(self, *argd)

def dumpProfile():
    templ = " %-40s | %8d | %8.2f | %8.2f | %8.2f\n"
    res  = " node name                                | ncalls   | avg time | total    | max time\n"
    res += "------------------------------------------+----------+----------+----------+---------\n"
    names = _profileNodes.keys()
    names.sort()
    for name in names:
        node_data = _profileNodes[name]
        level = name.count('.')
        name = "  " * level + name.split('.')[-1]
        total = node_data['total']
        ncalls = node_data['ncalls']
        max_ = node_data['max']
        avg = node_data['avg']
        res += templ % (name, ncalls, avg, total, max_)
    return res


def saveProfileLogs(fn):
    logs = {}
    for name in _profileNodes:
        node_data = _profileNodes[name]
        if 'log' in node_data:
            a = array(node_data['log'])
            logs[name] = a
    savez(fn, **logs)

def with_(*context):
    def wrap(func):
        def f(*args, **kargs):
            with ctx(*context):
                return func(*args, **kargs)
        return f
    return wrap

def setattrs(obj, **args):
    for name in args:
        setattr(obj, name, args[name])
    return obj


def genericFP(inline_code, profile = 'fp40'):
    uniforms = set( re.findall("(([a-z0-9]+)_\w+)", inline_code) )

    types = dict(f = 'float', f2 = 'float2', f3 = 'float3', f4 = 'float4', 
      s = 'sampler2D', s1 = 'sampler1D', s2 = 'sampler2D', s3 = 'sampler3D')

    uniforms = ['uniform %s %s;\n' % (types[t], n) for n, t in uniforms if t in types]
    uniforms = "".join(uniforms)

    if 'return' not in inline_code:
        inline_code = 'return ' + inline_code
    code = '''
      %s

      const float pi = 3.14159265359f;

      float4 main( 
        float4 tc0: TEXCOORD0,
        float4 tc1: TEXCOORD1) : COLOR 
      { 
        %s; 
      }
    ''' % (uniforms, inline_code)
    return CGShader(profile, code)

'''
from __future__ import with_statement
from zgl import *
    
class App(ZglAppWX):
    def __init__(self):
        ZglAppWX.__init__(self, viewControl = OrthoCamera())
        self.fragProg = genericFP('tc0')
    
    def display(self):
        clearGLBuffers()
        with ctx(self.viewControl.with_vp, self.fragProg):
            drawQuad()

if __name__ == "__main__":
    App().run()
'''