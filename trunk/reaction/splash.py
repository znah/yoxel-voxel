from __future__ import with_statement
from zgl import *

class App(ZglApp):
    def __init__(self):
        ZglApp.__init__(self, FlyCamera())
        self.fragProg = CGShader('fp40', TestShaders, entry = 'TexCoordFP')
        self.viewControl.eye = (0, 0, 10)
        self.viewControl.speed = 5

        n = 256
        self.verts = zeros((n, n, 2), float32)
        self.verts[...,1], self.verts[...,0] = indices((n, n))
        self.verts /= (n-1)
        
        idxgrid = arange(n*n).reshape(n, n)
        self.idxs= zeros((n-1, n-1, 4), uint32)
        self.idxs[...,0] = idxgrid[ :-1, :-1 ]
        self.idxs[...,1] = idxgrid[ :-1,1:   ]  
        self.idxs[...,2] = idxgrid[1:  ,1:   ]
        self.idxs[...,3] = idxgrid[1:  , :-1 ]
        self.idxs = self.idxs.flatten()

        self.vertProg = CGShader("vp40", '''
          uniform float time;

          uniform float Age;
          uniform float Intensity;
          uniform float SplashRadius;
          uniform float Dissipation;
          uniform float WavePhaseSpeed;
          uniform float Wavelength;
          
          const float pi = 3.141593;

          float splashHeight(float2 p)
          {
            if (Age < 0)
              return 0;
            float r = length(p);
            float wave = sin(2*pi*r/Wavelength - WavePhaseSpeed*Age);
            float timeFade = exp(-Dissipation*Age);
            float distFade = 1.0 / (r*r);
            float h = Intensity * timeFade * distFade;
            return min(h, 10.0);
          }

          void main(float2 inP : ATTR0, 
            
            out float4 oPos : POSITION,
            out float4 oTC : TEXCOORD0,
            out float3 oNormal : TEXCOORD1)
          {
            float3 pos = float3(inP*100, 0);
            float2 r = pos.xy - float2(50, 50);
            float h = splashHeight(r);

            pos.z = h;

            oPos = mul(glstate.matrix.mvp, float4(pos, 1));
            
            float v = h/6 + 0.5;
            oTC = float4(float3(v), 0);
            oNormal = float3(0, 0, 1);
          }
        ''')

        self.startTime = 0

    
    def display(self):
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glVertexAttribPointer(0, 2, GL_FLOAT, False, 0, self.verts.ctypes.data)
        
        self.vertProg.Age = self.time - self.startTime
        self.vertProg.Intensity = 100.0;
        self.vertProg.SplashRadius = 50;
        self.vertProg.Dissipation = 1.66;
        self.vertProg.WavePhaseSpeed = 10;
        self.vertProg.Wavelength = 5;

        with ctx(self.viewControl.with_vp, self.vertProg, self.fragProg, vattr(0), glstate(GL_DEPTH_TEST)):
            glDrawElements(GL_QUADS, len(self.idxs), GL_UNSIGNED_INT, self.idxs)
            #drawQuad()

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
