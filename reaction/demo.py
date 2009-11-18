from __future__ import with_statement
from numpy import *
from zgl import *
import time

viewSize = (640, 640)



def init():
    global colorFrag
    colorFrag = CGShader("fp40", """
      float4 main(float4 color : COLOR0) : COLOR { return color; }
    """)

    global texFrag
    texFrag = CGShader("fp40", """
      uniform sampler2D texture;
      float4 main(float2 texCoord : TEXCOORD0) : COLOR { return tex2D(texture, texCoord); }
    """)

    global reactFrag
    reactFrag = CGShader("fp40", """
      uniform sampler2D texture;
      uniform float2 dpos;
      uniform float fc;
      uniform float kc;
      float4 main(float2 pos : TEXCOORD0) : COLOR
      {
        float cdist = length(pos-float2(0.5, 0.5));
        float flow = 0.99;//length(pos-float2(0.5, 0.5)) > 0.2 ? 0.8 : 1.0;

        float2 v = tex2D(texture, pos).xy;
        float2 l = -4.0 * v;
        l += tex2D(texture, pos + float2( dpos.x, 0)).xy;
        l += tex2D(texture, pos + float2(-dpos.x, 0)).xy;
        l += tex2D(texture, pos + float2( 0, dpos.y)).xy*flow;
        l += tex2D(texture, pos + float2( 0,-dpos.y)).xy*(1/flow);
        
        if (cdist < 0.2)
          l *= 1 + (1-cdist/0.2) * 2;

        const float2 diffCoef = float2(0.082, 0.041*1.8);

        const float fs = 0.0;
        const float ks = 0.0;
        float d = saturate( length(2*pos-float2(1, 1)) );

        const float f = fc;//lerp(fc, 0.011, d);
        const float k = kc;//lerp(kc, 0.055, d);
        const float dt = 0.3f;

        float2 dv = diffCoef * l;
        float rate = v.x * v.y * v.y;
        dv += float2(-rate, rate);
        dv += float2(f * (1.0 - v.x), -(f + k) * v.y );
        v += dt * dv;
        return float4(v, rate, 0);
      }
    """)


    # f= 0.029  k = 0.055     diffCoef = float2(0.082, 0.041*1.8); !!!

    global fc, kc
    fc, kc = 0.027, 0.054
    reactFrag.fc = fc
    reactFrag.kc = kc
    global mouseX, mouseY
    mouseX, mouseY = (0, 0)

    global visFrag
    visFrag = CGShader("fp40", """
      uniform sampler2D texture;
      float4 main(float2 texCoord : TEXCOORD0) : COLOR 
      {
        float4 d = tex2D(texture, texCoord);
        return float4(30*d.z, d.y, d.x*0.5, 1); 
        //float v = d.y > 0.2 ? 1 : 0;
        //return float4(v, v, v, 1); 
      }
    """)

    global pp;
    a = zeros((512, 512, 4), float32)
    a[...,0] = 1
    a[...,1] = 0
    pp = PingPong(img = a, format = GL_RGBA_FLOAT32_ATI)
    pp.texparams((GL_TEXTURE_MAG_FILTER, GL_LINEAR))
    reactFrag.dpos = (1.0 / pp.size()[0], 1.0 / pp.size()[1])

    glEnable(GL_POINT_SMOOTH)

    global dropping

    global step
    step = 0.001

    global count, slices
    count = 0
    slices = zeros((512, a.shape[0]), float32)


def display():

    glutSetWindow(window);

    ipf = 100
    glFinish()
    startTime = time.clock()
    for i in xrange(ipf):
        reactFrag.texture = pp.src.tex
        with ctx(pp.dst, reactFrag, ortho):
            drawQuad()
        pp.flip()
    glFinish()
    dt = time.clock() - startTime
    iterTime = float(dt) / ipf * 1000

    '''
    global count
    if count < slices.shape[0]:
        with pp.src.tex:
            a = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)
        slices[count] = a[...,2].sum(1)
        count += 1
    if count == 512:
        save("slices.npy", slices)
        count += 1
    '''

    
    #iterCount += 100

    glViewport(0, 0, viewSize[0], viewSize[1])
    visFrag.texture = pp.src.tex
    with ortho:
        with visFrag:
            drawQuad()
        glWindowPos2i(100, 100);
        glColor(1, 1, 1, 1)
        glutBitmapString(GLUT_BITMAP_9_BY_15, "iterTime: %.3f ms  f = %f  k = %f, c: %d" % (iterTime, fc, kc, count));
    

    glutSwapBuffers()

def resize(x, y):
    global viewSize
    viewSize = (x, y)

def idle():
    glutPostRedisplay()

def keyDown(key, x, y):
    if ord(key) == 27:
        glutLeaveMainLoop()
    global fc, kc, step

    if key == 'a':
        fc -= step
    if key == 'd':
        fc += step
    if key == 's':
        kc -= step
    if key == 'w':
        kc += step
    if key == '[':
        step /= 2.0
    if key == ']':
        step *= 2.0
    reactFrag.fc = fc
    reactFrag.kc = kc

def dropDrop(x, y):
    with ctx(pp.src, colorFrag, ortho):
        glColor4f(0.5, 0.25, 0, 0)
        glPointSize(10.0)
        glBegin(GL_POINTS)
        glVertex(float(x) / viewSize[0], 1 - float(y) / viewSize[1])
        glEnd()

def mouseButton(btn, up, x, y):
    if up:
        dropping = False
        return
    dropDrop(x, y)
    dropping = True

def mouseMotion(x, y):
    dropDrop(x, y)
    


if __name__ == "__main__":
  glutInit([])
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

  glutInitWindowSize(viewSize[0], viewSize[1])

  window = glutCreateWindow("hello")
  glutDisplayFunc(display)
  glutIdleFunc(idle)
  glutKeyboardFunc(keyDown)
  #glutKeyboardUpFunc(keyUp)
  glutMouseFunc(mouseButton)
  glutMotionFunc(mouseMotion)
  glutReshapeFunc(resize)
  #glutCloseFunc(close)

  InitCG()
  init()


  #wglSwapIntervalEXT(0)
  glutSetOption ( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION )
  glutMainLoop()
  
