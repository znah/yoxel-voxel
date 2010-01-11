from __future__ import with_statement
from zgl import *

def sample_ball(n):
    a = zeros((0, 3), float32)    
    while len(a) < n:
        b = random.rand(n - len(a), 3) * 2.0 - 1.0
        v = sum(b*b, axis=1)
        b = b[v<1.0]
        a = vstack((a, b))
    return a.astype(float32)   
        
class Explosion:
    def __init__(self):
        bb = array([[[0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [1, 1, 1]]], float32)
        gradTex = Texture2D(img = bb)
        gradTex.filterLinear()
        gradTex.setParams(*Texture.ClampToEdge)
    
        self.spriteFP = CGShader('fp40', '''
          uniform sampler2D tex;
          uniform sampler2D gradTex;
          float4 main(float2 tc: TEXCOORD0, float fade: TEXCOORD1) : COLOR
          {
            float4 c = tex2D(tex, tc);
            c.rgb = tex2D(gradTex, float2(c.a*0.8, 0.5)).rgb;
            c.a *= fade;
            return c;
          }
        ''')
        self.spriteFP.tex = loadTex('img\\smoke.png')
        self.spriteFP.gradTex = gradTex
        
        self.spriteVP = CGShader('vp40', '''
          uniform float age;
          
          uniform float3 basePos;
          
          void main( 
            float3 inPos  : ATTR0,
            float3 inVel  : ATTR1,
            float2 inTC   : ATTR2,
            float2 inRotation : ATTR4, // (ang0, angVel)
            out float4 oPos  : POSITION,
            out float4 oTC   : TEXCOORD0,
            out float oFade : TEXCOORD1) 
          {
            float4 posWld = float4(basePos + inPos, 1.0);
            float4 posEye = mul(glstate.matrix.modelview[0], posWld);
            
            float ang = inRotation.x + inRotation.y * age;
            float size = 10.0 * (1 - exp(-age));
            
            float2 u;
            sincos(ang, u.y, u.x);
            float2 v = float2(-u.y, u.x);
            float2 uv = (inTC-0.5) * size;
            
            posEye.xy += uv.x * u + uv.y * v;
            oPos = mul(glstate.matrix.projection, posEye);
            oTC = float4(inTC, 0, 1);
            
            oFade = 1;//smoothstep(0.0, 5.0, age);
            
          }
        ''')

        self.spriteVP.basePos = (50, 50, 10)
        
        self.reset(0.0)
        
    def reset(self, startTime):
        self.startTime = startTime
        self.lastTime = startTime

        self.pNum = 100
        self.ppos = sample_ball(self.pNum) * 1.0
        
        self.pvel = sample_ball(self.pNum) * 30.0
        self.zorder = arange(self.pNum).astype(uint32)
        
        self.arrTC = tile([[0, 0], [1, 0], [1, 1], [0, 1]], (self.pNum, 1)).astype(float32)
        
        rotation = random.rand(self.pNum, 2).astype(float32)
        ang0 = rotation[:,0]
        angVel = rotation[:,1]
        ang0 *= 2*pi
        angVel[:] = angVel*2 - 1
        #angVel *= 5.0
        self.arrRot = tile(rotation, (1, 4))
        
        self.updateRenderArrays()
        
    def update(self, time, viewControl):
        dt = time - self.lastTime
        self.lastTime = time
        self.age = time - self.startTime
        if self.age < 0:
            return
            
        self.ppos += self.pvel * dt
        
        eyeDir = viewControl.forwardVec()
        z = dot(self.ppos, eyeDir)
        self.zorder = argsort(-z).astype(uint32)
        
        self.updateRenderArrays()
        
    def updateRenderArrays(self, indexOnly = False):
        if not indexOnly:
            self.arrPos = tile(self.ppos,  (1, 4))
            self.arrVel = tile(self.pvel,  (1, 4))
        self.arrIdx = repeat(self.zorder*4, 4).reshape(-1, 4)
        self.arrIdx += [0, 1, 2, 3]
        
    def render(self):
        if self.age < 0:
            return
            
        glDepthMask(GL_FALSE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                  
        self.spriteVP.age = self.age
                        
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, self.arrPos.ctypes.data)
        glVertexAttribPointer(1, 3, GL_FLOAT, False, 0, self.arrVel.ctypes.data)
        glVertexAttribPointer(2, 2, GL_FLOAT, False, 0, self.arrTC.ctypes.data)
        glVertexAttribPointer(4, 2, GL_FLOAT, False, 0, self.arrRot.ctypes.data)
        flags = [GL_DEPTH_TEST, GL_BLEND]
        with ctx(self.spriteVP, self.spriteFP, vattr(0, 1, 2, 4), glstate(*flags)):
            glDrawElements(GL_QUADS, self.pNum * 4, GL_UNSIGNED_INT, self.arrIdx)
            
        glDepthMask(GL_TRUE)    


class App(ZglApp):
    def __init__(self):
        ZglApp.__init__(self, FlyCamera())
        self.fragProg = CGShader('fp40', TestShaders, entry = 'TexCoordFP')
        self.viewControl.eye = (0, 0, 10)
        self.viewControl.speed = 5

        self.vertProg = CGShader("vp40", '''
          #line 13
          uniform float time;
          
          const float pi = 3.141593;

          float height(float2 p)
          {
            float h = sin(p.x+time*2) + sin(0.5*p.y+time);
            return 0.3*h;
          }

          void main(float2 inP : ATTR0, 
            
            out float4 oPos : POSITION,
            out float4 oTC : TEXCOORD0,
            out float3 oNormal : TEXCOORD1)
          {
            float2 xy = float2(inP*100);
            float z = height(xy);

            float step = 0.1;
            float dx = height(xy + float2(step, 0)) - z;
            float dy = height(xy + float2(0, step)) - z;
            float3 n = normalize(float3(-dx, -dy, step));
            const float3 lightDir = normalize(float3(-1 ,-1, 1));
            float diff = max(dot(lightDir, n), 0.0);

            oPos = mul(glstate.matrix.mvp, float4(xy, z, 1));

            float3 col =float3(0.1, 0.4, 0.6);

            oTC = float4(col*(0.1+diff*0.9), 0);
            oNormal = float3(0, 0, 1);
          }
        ''')
        
        self.exlposion = Explosion()

    
    def display(self):
        glClearColor(0, 0.5, 0.7, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self.vertProg.time = self.time
        self.exlposion.update(self.time, self.viewControl)

        with self.viewControl.with_vp:
            with ctx(self.vertProg, self.fragProg, glstate(GL_DEPTH_TEST)):
                drawGrid(256)
            self.exlposion.render()    

        glutSwapBuffers()

    def keyDown(self, key, x, y):
        if key == ' ':
            self.exlposion.reset(self.time)
        else:
            ZglApp.keyDown(self, key, x, y)


if __name__ == "__main__":
    viewSize = (800, 600)
    zglInit(viewSize, "hello")

    #wglSwapIntervalEXT(0)
    glutSetCallbacks(App())
    glutMainLoop()
