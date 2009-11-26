from __future__ import with_statement
from zgl import *

class App(ZglApp):
    def __init__(self):
        ZglApp.__init__(self, FlyCamera())

        data = fromfile("img/bonsai.raw", uint8)
        data.shape = (256, 256, 256)
        self.volumeTex =  Texture3D(img=data)
        self.volumeTex.filterLinear()
        self.traceVP = CGShader('vp40', '''
          uniform float3 eyePos;
          float4 main(float2 p : POSITION, 
             out float3 rayDir : TExCOORD0 ) : POSITION
          {
            float4 projVec = float4(p * 2.0 - float2(1.0, 1.0), 0, 1);
            float4 wldVec = mul(glstate.matrix.inverse.mvp, projVec);
            wldVec /= wldVec.w;
            rayDir = wldVec.xyz - eyePos;
            return projVec;
          }
        
        ''')

        self.traceFP = CGShader('fp40', '''
        # line 26
          uniform sampler3D volume;
          uniform float time;
          uniform float3 eyePos;

          void hitBox(float3 o, float3 d, out float tenter, out float texit)
          {
            float3 invD = 1.0 / d;
            float3 tlo = invD*(-o);
            float3 thi = invD*(float3(1, 1, 1)-o);

            float3 t1 = min(tlo, thi);
            float3 t2 = max(tlo, thi);

            tenter = max(t1.x, max(t1.y, t1.z));
            texit  = min(t2.x, min(t2.y, t2.z));
          }

          float4 main( float3 rayDir: TEXCOORD0 ) : COLOR 
          {
            rayDir = normalize(rayDir);
            float t1, t2;
            hitBox(eyePos, rayDir, t1, t2);
            t1 = max(0, t1);
            if (t1 > t2)
              return float4(0, 0, 0, 0);
            float3 p = eyePos + rayDir * t1;

            float dt = 0.01;
            float3 step = dt * rayDir;
            for (float t = t1; t < t2; t += dt)
            {
              float v = tex3D(volume, p);
              if (v > 0.5)
                return float4(p, 1);
              p += step;
            }
            return float4(0, 0, 0, 1);
          }
        ''')
        self.traceFP.volume = self.volumeTex
        self.viewControl.eye = (0, -1, 1)

    def display(self):
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.traceVP.eyePos = self.viewControl.eye
        self.traceFP.eyePos = self.viewControl.eye
        self.traceFP.time = self.time
        with ctx(self.viewControl.with_vp, self.traceVP, self.traceFP):
            drawQuad()

        glutSwapBuffers()

if __name__ == "__main__":
    viewSize = (800, 600)
    zglInit(viewSize, "hello")

    glutSetCallbacks(App())

    #wglSwapIntervalEXT(0)
    glutMainLoop()
