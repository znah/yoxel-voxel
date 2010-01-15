from __future__ import with_statement
from zgl import *

class App(ZglAppWX):
    def __init__(self):
        ZglAppWX.__init__(self, viewControl = FlyCamera())

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

        self.traceFP = CGShader('gp4fp', '''
        # line 30
          uniform sampler3D volume;
          uniform float time;
          uniform float3 eyePos;
          uniform float dt;
         
          void hitBox(float3 o, float3 d, out float tenter, out float texit)
          {
            float3 invD = 1.0 / d;
            float3 tlo = invD*(-o);
            float3 thi = invD*(float3(1.0, 1.0, 1.0)-o);

            float3 t1 = min(tlo, thi);
            float3 t2 = max(tlo, thi);

            tenter = max(t1.x, max(t1.y, t1.z));
            texit  = min(t2.x, min(t2.y, t2.z));
          }

          float3 getnormal(float3 p)
          {
            float d = 0.2/256.0;
            float x = tex3D(volume, p + float3(d, 0, 0)).r - tex3D(volume, p - float3(d, 0, 0)).r;
            float y = tex3D(volume, p + float3(0, d, 0)).r - tex3D(volume, p - float3(0, d, 0)).r;
            float z = tex3D(volume, p + float3(0, 0, d)).r - tex3D(volume, p - float3(0, 0, d)).r;
            float3 n = -0.5*float3(x, y, z);
            return normalize(n);
          }

          float4 main( float3 rayDir: TEXCOORD0 ) : COLOR 
          {
            const float3 lightDir = normalize(float3(1, 1, 1));

            rayDir = normalize(rayDir);
            float t1, t2;
            hitBox(eyePos, rayDir, t1, t2);
            t1 = max(0.0, t1);
            if (t1 > t2)
              return float4(0, 0, 0, 0);
            float3 p = eyePos + rayDir * t1;

            float3 step = dt * rayDir;
            float4 res = float4(0);
            float c0 = tex3D(volume, p).r;
            p += step;

            const float ths = 0.2;
            for (float t = t1+dt; t < t2; t += dt, p += step)
            {
              float c1 = tex3D(volume, p).r;
              if (c0 < ths && c1 > ths)
              {
                float3 p1 = p-step, p2 = p;
                for (int i = 0; i < 4; ++i)
                {
                  float r = (ths - c0) / (c1 - c0);
                  float3 pm = p1 + (p2-p1)*r;
                  float cm = tex3D(volume, pm).r;
                  if (cm < ths)
                  {
                    p1 = pm;
                    c0 = cm;
                  }
                  else
                  {
                    p2 = pm;
                    c1 = cm;
                  }
                }
                float3 hitP = 0.5*(p1 + p2);
                float3 n = getnormal(hitP);
                float diffuse = dot(n, lightDir);
                res = float4(diffuse, diffuse, diffuse, 1.0);
                break;
              }
              c0 = c1;
            }
            return res;
          }
        ''')
        self.traceFP.volume = self.volumeTex
        self.viewControl.eye = (0, -1, 1)
        self.traceFP.dt = 0.01

    def display(self):
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.traceVP.eyePos = self.viewControl.eye
        self.traceFP.eyePos = self.viewControl.eye
        self.traceFP.time = self.time
        with ctx(self.viewControl.with_vp, self.traceVP, self.traceFP):
            drawQuad()

    def OnKeyDown(self, evt):
        key = evt.GetKeyCode()
        if key == ord('['):
            self.traceFP.dt = 0.5 * self.traceFP.dt
        if key == ord(']'):
            self.traceFP.dt = 2.0 * self.traceFP.dt
        else:
            ZglAppWX.OnKeyDown(self, evt)


if __name__ == "__main__":
    App().run()
