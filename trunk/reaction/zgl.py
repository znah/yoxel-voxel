from __future__ import with_statement
import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.EXT.framebuffer_object import *
from OpenGL.GL.EXT.texture_integer import *
from numpy import *

# freeglut hack
platform.GLUT = ctypes.windll.LoadLibrary("freeglut")
from OpenGL.GLUT import *
special._base_glutInit = OpenGL.raw.GLUT.glutInit
glutCreateWindow = OpenGL.raw.GLUT.glutCreateWindow
glutCreateMenu = OpenGL.raw.GLUT.glutCreateMenu

from ctypes import cdll, c_int, c_uint, c_float, c_char_p
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

cgGLSetTextureParameter = cggl.cgGLSetTextureParameter
cgGLSetTextureParameter.argtypes = [c_int, c_uint]

cgProfiles = {"fp40"  : 6151, 
              "vp40"  : 7001, 
              "gp4fp" : 7010,
              "gp4vp" : 7011,
              "gp4gp" : 7012}

cgParamSetters = {
  cg.cgGetType("float")     : lambda p, v : cgGLSetParameter1f(p, v),
  cg.cgGetType("float2")    : lambda p, v : cgGLSetParameter2f(p, v[0], v[1]),
  cg.cgGetType("float3")    : lambda p, v : cgGLSetParameter3f(p, v[0], v[1], v[2]),
  cg.cgGetType("float4")    : lambda p, v : cgGLSetParameter4f(p, v[0], v[1], v[2], v[3]),
  cg.cgGetType("sampler2D") : lambda p, v : cgGLSetTextureParameter(p, v)
}


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
    raise Exception(msg, listing)


'''
class Vec2:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y
    def __add__(self, other)
        return Vec2(self.x + other.x, self.y + other.y)

def XY(v):
    return Vec2(v[0], v[1])
def YX(v):
    return Vec2(v[1], v[0])
'''

def V(*args):
    if len(args) == 1:
        return array(args[0], float32)
    else:
        return array(args, float32)

def YX(*args):
    if len(args) == 1:
        return array((args[0][1], args[0][0]), float32)
    else:
        return array((args[1], args[0]), float32)
                            
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

class Texture2D:
    ChNum2Format = {1:GL_LUMINANCE, 3:GL_RGB, 4:GL_RGBA}
    
    Nearest = [(GL_TEXTURE_MIN_FILTER, GL_NEAREST), (GL_TEXTURE_MAG_FILTER, GL_NEAREST)]
    Linear  = [(GL_TEXTURE_MIN_FILTER, GL_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR)]
    MipmapLinear  = [(GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR)]
    Repeat  = [(GL_TEXTURE_WRAP_S, GL_REPEAT), (GL_TEXTURE_WRAP_S, GL_REPEAT)]

    def __init__(self, img = None, size = None, format = GL_RGBA8, srcFormat = GL_RGBA, srcType = GL_FLOAT):
        self._as_parameter_ = glGenTextures(1)
        self.setParams( *(self.Nearest + self.Repeat) )
        if img != None:
            with self:
                img = atleast_3d(ascontiguousarray(img))
                srcFormat = self.ChNum2Format[img.shape[2]]
                srcType = arrays.ArrayDatatype.getHandler(img).arrayToGLType(img)
                glPixelStorei(GL_PACK_ALIGNMENT, 1);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                glTexImage2D(GL_TEXTURE_2D, 0, format, img.shape[1], img.shape[0], 0, srcFormat, srcType, img)
                self.size = YX(img.shape[:2])
            return
        elif shape != None:
            with self:
                glTexImage2D(GL_TEXTURE_2D, 0, format, size[0], size[1], 0, srcFormat, srcType, None)
                self.size = size

    def setParams(self, *args):
        with self:
            for pname, val in args:
                glTexParameteri(GL_TEXTURE_2D, pname, val)

    def genMipmaps(self):
        with self:
            glGenerateMipmapEXT(GL_TEXTURE_2D)
    
    def __enter__(self):
        glBindTexture(GL_TEXTURE_2D, self)

    def __exit__(self, *args):
        glBindTexture(GL_TEXTURE_2D, 0)


class RenderTexture:
    def __init__(self, depth = False, **args):
        self.fbo = glGenFramebuffersEXT(1)
        self.tex = Texture2D(**args)
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, self.fbo)
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, self.tex, 0)
        if depth:
            self.depthRB = glGenRenderbuffersEXT(1)
            glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, self.depthRB)
            glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, self.size()[0], self.size()[1])
            glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, self.depthRB)
            glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0)
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0)

    def size(self):
        return self.tex.size

    def __enter__(self):
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, self.fbo)
        glViewport(0, 0, self.size()[0], self.size()[1])

    def __exit__(self, *args):
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0)

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

_quad = [(0, 0), (1, 0), (1, 1), (0, 1)]
def drawQuad():
    glBegin(GL_QUADS)
    def vert(p):
        glTexCoord2f(p[0], p[1])
        glVertex2f(p[0], p[1])
    [vert(p) for p in _quad]
    glEnd()

def drawVerts(primitive, pos, texCoord = None):
    glBegin(primitive)
    for i in xrange(len(pos)):
        if texCoord is not None: 
            glTexCoord(*texCoord[i])
        glVertex(*pos[i])
    glEnd()
    


def glutSetCallbacks(app):
    if hasattr(app, "display"):
        glutDisplayFunc(app.display)
    if hasattr(app, "resize"):
        glutReshapeFunc(app.resize)
    if hasattr(app, "idle"):
        glutIdleFunc(app.idle)
    if hasattr(app, "keyDown"):
        glutKeyboardFunc(app.keyDown)
    if hasattr(app, "keyUp"):
        glutKeyboardUpFunc(app.keyUp)
    if hasattr(app, "mouseMove"):
        glutMotionFunc(app.mouseMove)
    if hasattr(app, "mouseButton"):
        glutMouseFunc(app.mouseButton)
    if hasattr(app, "close"):
        glutCloseFunc(app.close)

class FlyCamera:
    def __init__(self):
        self.eye = (0.0, 0.0, 0.0)
        self.course = 0
        self.pitch = 0
        self.fovy = 40
        self.viewSize = (1, 1)
        self.zNear = 1.0
        self.zFar  = 1000.0
        
        self.mButtons = zeros((3,), bool)
        self.mPos = (0, 0)
        self.keyModifiers = 0
        
        self.vel = [0, 0]
        self.sensitivity = 0.3
        self.speed = 1.0

    def __setattr__(self, name, val):
        if name == "eye":
            val = V(*val)
        self.__dict__[name] = val

    def resize(self, x, y):
        self.viewSize = (x, max(y, 1))

    key2vel = {'w': (0, 1), 's': (0, -1), 'd': (1, 1), 'a': (1, -1) }

    def keyDown(self, key, x, y):
        self.keyModifiers = glutGetModifiers()
        k = key.lower()
        if self.key2vel.has_key(k):
            a, d = self.key2vel[k]
            self.vel[a] = d

    def keyUp(self, key, x, y):
        self.keyModifiers = glutGetModifiers()
        k = key.lower()
        if self.key2vel.has_key(k):
            a, d = self.key2vel[k]
            self.vel[a] = 0

    def mouseMove(self, x, y):
        dx = x - self.mPos[0]
        dy = y - self.mPos[1]
        if self.mButtons[GLUT_LEFT_BUTTON]:
            self.course -= dx * self.sensitivity
            self.pitch = clip(self.pitch - dy * self.sensitivity, -89.9, 89.9)
        self.mPos = (x, y)
        self.keyModifiers = glutGetModifiers()

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
        self.keyModifiers = glutGetModifiers()

    def __enter__(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = float(self.viewSize[0]) / self.viewSize[1]
        gluPerspective(self.fovy, aspect, self.zNear, self.zFar)
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
    
    def updatePos(self, dt):
        v = self.vel[0] * self.forwardVec() + self.vel[1] * self.rightVec()
        if self.keyModifiers & GLUT_ACTIVE_SHIFT != 0:
            v *= 10
        self.eye += v*self.speed*dt

def zglInit(viewSize, title):
    glutInit([])
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(*viewSize)
    glutCreateWindow(title)
    glutSetOption ( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION )
    InitCG()

class BufferObject:
    def __init__(self):
        self._as_parameter_ = glGenBuffers(1)
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


class OrthoCamera:
    def __init__(self, viewSize = (1, 1)):
        self.viewSize = V(viewSize)
        self.center = V(0.5, 0.5)
        self.scaley = 1.0

        self.mButtons = zeros((3,), bool)
        self.mPos = (0, 0)
        self.keyModifiers = 0
   
    def extent(self):
        aspect = float(self.viewSize[0]) / self.viewSize[1]
        return V(aspect * self.scaley, self.scaley)

    def rect(self):
        halfExt = 0.5 * self.extent()
        lo = self.center - halfExt
        hi = self.center + halfExt
        return (lo[0], lo[1], hi[0], hi[1])

    def resize(self, x, y):
        self.viewSize = V(max(x, 1), max(y, 1))

    def mouseMove(self, x, y):
        dx = x - self.mPos[0]
        dy = y - self.mPos[1]
        if self.mButtons[GLUT_LEFT_BUTTON]:
            self.center += V(-dx, dy) / self.viewSize * self.extent()
        self.mPos = (x, y)
        self.keyModifiers = glutGetModifiers()

    def scale(self, coef):
        self.scaley *= coef

    def mouseButton(self, btn, up, x, y):
        self.mPos = (x, y)
        self.keyModifiers = glutGetModifiers()
        if btn < 3:
            self.mButtons[btn] = not up
        if btn == 3:
            self.scale(1.0/1.1)
        if btn == 4:
            self.scale(1.1)

    def __enter__(self):
        self.proj = Ortho(self.rect())
        self.proj.__enter__()
    def __exit__(self, *args):
        self.proj.__exit__()

'''
from __future__ import with_statement
from zgl import *


class App:
    def __init__(self, viewSize):
        self.viewControl = OrthoCamera()

    def resize(self, x, y):
        self.viewControl.resize(x, y)

    def idle(self):
        glutPostRedisplay()
    
    def display(self):
        
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glViewport(0, 0, self.viewControl.viewSize[0], self.viewControl.viewSize[1])

        with self.viewControl:
            drawQuad()

        glutSwapBuffers()

    def keyDown(self, key, x, y):
        if ord(key) == 27:
            glutLeaveMainLoop()
                
    def keyUp(self, key, x, y):
        pass

    def mouseMove(self, x, y):
        self.viewControl.mouseMove(x, y)

    def mouseButton(self, btn, up, x, y):
        self.viewControl.mouseButton(btn, up, x, y)


if __name__ == "__main__":
  viewSize = (800, 600)
  zglInit(viewSize, "hello")

  app = App(viewSize)
  glutSetCallbacks(app)

  #wglSwapIntervalEXT(0)
  glutMainLoop()
'''