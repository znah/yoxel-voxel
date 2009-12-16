from __future__ import with_statement
from zgl import *

class App(ZglApp):
    def __init__(self):
        ZglApp.__init__(self, FlyCamera())
        self.fragProg = CGShader('fp40', TestShaders, entry = 'TexCoordFP')
        self.viewControl.eye = (0, 0, 10)
        self.viewControl.speed = 5

        self.vertProg = CGShader("vp40", '''
          #line 26
          uniform float time;

          uniform float Age;
          uniform float Intensity;
          uniform float SplashRadius;

          uniform float WaveFadeCoef;
          
          uniform float WaveVelocity;
          uniform float Wavelength;
          
          const float pi = 3.141593;

          float2 splashHeightRad(float r, float frontDist)
          {
            float waveFade = r < frontDist ? pow(WaveFadeCoef, frontDist - r) / (1+r) : 0;

            float shock = 5*exp(-0.5*r*r/frontDist/frontDist) / (1 + frontDist);
            shock = min(shock, 10);

            float wave = -sin(2*pi*(r-frontDist)/Wavelength);
            float h = Intensity * wave * waveFade + shock;
            return float2(h, shock/(1+Age));
          } 
          
          float2 splashHeight(float2 p)
          {
            float r = length(p);
            float frontDist = Age * WaveVelocity;
            return splashHeightRad(r, frontDist);
          }

          void main(float2 inP : ATTR0, 
            
            out float4 oPos : POSITION,
            out float4 oTC : TEXCOORD0,
            out float3 oNormal : TEXCOORD1)
          {
            float3 pos = float3(inP*40, 0);
            float2 r = pos.xy - float2(20, 20);
            float2 h_col = splashHeight(r);
            float h = h_col.x;
            float c = h_col.y;

            float step = 0.1;
            float dx = splashHeight(r + float2(step, 0)).x - h;
            float dy = splashHeight(r + float2(0, step)).x - h;
            float3 n = normalize(float3(-dx/step, -dy/step, 1));
            const float3 lightDir = normalize(float3(-1 ,-1, 1));
            float diff = max(dot(lightDir, n), 0.0);

            pos.z = h;

            oPos = mul(glstate.matrix.mvp, float4(pos, 1));
            
            float shockR = Age*50 + 0.1;
            float x = length(r)-shockR;
            float a = Age*50+0.1;
            c = exp(-x*x/a/a)/a;

            float3 col = lerp(float3(0.1, 0.2, 0.6), float3(1), c);

            oTC = float4(col*(0.1+diff*0.9), 0);
            oNormal = float3(0, 0, 1);
          }
        ''')

        self.startTime = 0

    
    def display(self):
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self.vertProg.Age = self.time - self.startTime
        self.vertProg.Intensity = 10.0;
        self.vertProg.WaveFadeCoef= 0.5;
        self.vertProg.WaveVelocity = 10;
        self.vertProg.Wavelength = 3;

        with ctx(self.viewControl.with_vp, self.vertProg, self.fragProg, glstate(GL_DEPTH_TEST)):
            drawGrid(256)

        glutSwapBuffers()

    def keyDown(self, key, x, y):
        if key == ' ':
            self.startTime = self.time
        else:
            ZglApp.keyDown(self, key, x, y)


if __name__ == "__main__":
    viewSize = (800, 600)
    zglInit(viewSize, "hello")

    glutSetCallbacks(App())

    #wglSwapIntervalEXT(0)
    glutMainLoop()
