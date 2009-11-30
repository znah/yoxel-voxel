from __future__ import with_statement
from zgl import *

class App(ZglApp):
    def __init__(self):
        ZglApp.__init__(self, FlyCamera())

        data = fromfile("img/bonsai.raw", uint8)
        data.shape = (256, 256, 256)
        data = swapaxes(data, 0, 1)
        self.volumeTex =  Texture3D(img=data)
        self.volumeTex.filterLinear()
        self.volumeTex.setParams(*Texture.Clamp)
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
        # line 30
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

          float3 getnormal(float3 p)
          {
            float d = 1.0/256.0;
            float x = tex3D(volume, p + float3(d, 0, 0)) - tex3D(volume, p - float3(d, 0, 0));
            float y = tex3D(volume, p + float3(0, d, 0)) - tex3D(volume, p - float3(0, d, 0));
            float z = tex3D(volume, p + float3(0, 0, d)) - tex3D(volume, p - float3(0, 0, d));
            float3 n = -0.5*float3(x, y, z);
            return normalize(n);
          }

          float4 main( float3 rayDir: TEXCOORD0 ) : COLOR 
          {
            const float3 lightDir = normalize(float3(1, 1, 1));

            rayDir = normalize(rayDir);
            float t1, t2;
            hitBox(eyePos, rayDir, t1, t2);
            t1 = max(0, t1);
            if (t1 > t2)
              return float4(0, 0, 0, 0);
            float3 p = eyePos + rayDir * t1;

            float dt = 0.005;
            float3 step = dt * rayDir;
            float4 res = float4(0);
            for (float t = t1; t < t2; t += dt, p += step)
            {
              float v = tex3D(volume, p);
              v = (v - 0.2) * 10;
              if (v < 0.0)
                continue;
              v = saturate(v);

              float3 n = getnormal(p);
              float diff = max(dot(lightDir, n), 0);
              float4 col = float4(diff*v, diff*v, diff*v, v);
              float trans = 1.0f - res.a;
              res.rgb += col.rgb * trans;
              res.a += trans * col.a;
            }
            return res;
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
