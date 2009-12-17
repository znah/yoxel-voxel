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
        self.spriteFP = CGShader('fp40', '''
          //uniform sampler2D tex;
          float4 main(float2 tc: TEXCOORD0) : COLOR
          {
            float2 p = 2.0 * tc - 1.0;
            float r = length(p);
            float a = smoothstep(1.0, 0.0, r);
            float3 c1 = float3(0.3, 0.2, 0.2);
            float3 c2 = float3(1.0, 0.7, 0.2);
            float3 c = lerp(c1, c2, a*a);
            return float4(c, a);//tex2D(tex, tc);
          }
        ''')
        
        self.spriteVP = CGShader('vp40', '''
          uniform float age;
          uniform float spriteSize;
          uniform float projCoef;
          
          uniform float3 basePos;
          
          void main( 
            float3 inPos  : ATTR0,
            float3 inVel  : ATTR1,
            out float4 oPos  : POSITION,
            out float oPSize : PSIZE) 
          {
            oPos = mul(glstate.matrix.mvp, float4(basePos + inPos, 1));
            oPSize = projCoef * spriteSize / oPos.w;
          }
        ''')

        self.spriteVP.spriteSize = 5.0
        self.spriteVP.basePos = (50, 50, 10)
        
        self.reset(0.0)
        
    def reset(self, startTime):
        self.startTime = startTime
        self.lastTime = startTime

        self.pNum = 1000
        self.ppos = sample_ball(self.pNum) * 1.0
        self.pvel = sample_ball(self.pNum) * 50.0
        self.zidx = arange(self.pNum).astype(uint32)

        
    def update(self, time, viewControl):
        dt = time - self.lastTime
        self.lastTime = time
        self.age = time - self.startTime
        if self.age < 0:
            return
            
        self.ppos += self.pvel * dt
        self.pvel *= 0.5**dt
        self.pvel += sin(self.ppos*0.1)*dt*10
        
        eyeDir = viewControl.forwardVec()
        z = dot(self.ppos, eyeDir)
        self.zidx = argsort(-z).astype(uint32)
        
        fov = radians(viewControl.fovy)
        h = viewControl.vp.size[1]
        self.spriteVP.projCoef = h / (2.0*tan(0.5*fov))
        
    def render(self):
        if self.age < 0:
            return
            
        glTexEnvf(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)
        glDepthMask(GL_FALSE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                        
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, self.ppos.ctypes.data)
        glVertexAttribPointer(1, 3, GL_FLOAT, False, 0, self.pvel.ctypes.data)
        flags = [GL_POINT_SPRITE, GL_VERTEX_PROGRAM_POINT_SIZE_ARB, GL_DEPTH_TEST, GL_BLEND]
        with ctx(self.spriteVP, self.spriteFP, vattr(0, 1), glstate(*flags)):
            glDrawElements(GL_POINTS, len(self.zidx), GL_UNSIGNED_INT, self.zidx)
            
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
