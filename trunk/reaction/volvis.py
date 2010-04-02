from __future__ import with_statement
from zgl import *

class VolumeRenderer(HasTraits):
    density        = Range( 0.0, 1.0, 0.5)
    brightness     = Range( 0.0, 2.0, 1.0 )
    transferOffset = Range(-1.0, 1.0, 0.0 )
    transferScale  = Range( 0.0, 2.0, 1.0 )
    stepsInTexel   = Range( 0.5, 4.0, 2.0 )

    transferFunc   = Enum('gray', 'nv')

    volumeTex      = Any(editable = False)
    
    _ = Python(editable = False)

    @on_trait_change( '+' )
    def updateShaderParams(self):
        if not hasattr(self, 'traceFP'):
            return
        self.traceFP.brightness = self.brightness
        self.traceFP.transferOffset = self.transferOffset
        self.traceFP.transferScale = self.transferScale
        self.traceFP.transferScale = self.transferScale
        self.traceFP.density = self.density / self.stepsInTexel
        self.traceFP.transferTex = self.trans[self.transferFunc]
        if self.volumeTex is not None:
            self.traceFP.dt = 1.0 / (self.volumeTex.size[0] * self.stepsInTexel)
            self.volumeTex.filterLinear()
            self.volumeTex.setParams(*Texture.Clamp)
            self.traceFP.volume = self.volumeTex
            
    
    def __init__(self, volumeTex = None):
        HasTraits.__init__(self)
        
        self.volumeTex = volumeTex
            
        def makeTransTex(a):
            tex = Texture1D(a, format = GL_RGBA8)
            tex.filterLinear()
            tex.setParams(*Texture.Clamp)
            return tex

        gray = makeTransTex(array([
            [  0.0, 0.0, 0.0, 0.0 ],
            [  1.0, 1.0, 1.0, 1.0 ]], float32))

        nv = makeTransTex(array([
            [  0.0, 0.0, 0.0, 0.0 ],
            [  1.0, 0.0, 0.0, 1.0 ],
            [  1.0, 0.5, 0.0, 1.0 ],
            [  1.0, 1.0, 0.0, 1.0 ],
            [  0.0, 1.0, 0.0, 1.0 ],
            [  0.0, 1.0, 1.0, 1.0 ],
            [  0.0, 0.0, 1.0, 1.0 ],
            [  1.0, 0.0, 1.0, 1.0 ],
            [  0.0, 0.0, 0.0, 0.0 ]], float32))

        self.trans = {'nv': nv, 'gray': gray}
            
        self.traceVP = CGShader('vp40', '''
          #line 58
          float4 main(float2 p : POSITION, 
             out float3 rayDir : TExCOORD0,
             out float3 eyePos : TExCOORD1) : POSITION
          {
            float4 projVec = float4(p * 2.0 - float2(1.0, 1.0), 0, 1);
            float4 wldVec = mul(glstate.matrix.inverse.mvp, projVec);
            eyePos = mul(glstate.matrix.inverse.modelview[0], float4(0, 0, 0, 1)).xyz;
            wldVec /= wldVec.w;
            rayDir = wldVec.xyz - eyePos;
            return projVec;
          }
        
        ''')

        self.traceFP = CGShader('gp4fp', '''
        # line 44
          uniform sampler1D transferTex;
          uniform sampler3D volume;
          
          uniform float dt;
          uniform float density;
          uniform float brightness;
          uniform float transferOffset;
          uniform float transferScale;
         
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

          float4 main( float3 rayDir: TEXCOORD0, float3 eyePos : TEXCOORD1) : COLOR 
          {
            rayDir = normalize(rayDir);
            float t1, t2;
            hitBox(eyePos, rayDir, t1, t2);
            t1 = max(0.0, t1);
            if (t1 > t2)
              return float4(0, 0, 0, 0);
            float3 p = eyePos + rayDir * t1;

            float step = dt;// * max(t1, 0.1);
            float4 accum = float4(0);
            for (float t = t1; t < t2; t += step, p += step * rayDir)
            {
              //step = dt * max(t1, 0.1);
              float sample = tex3D(volume, p).r;
              float4 col = tex1D(transferTex, (sample - transferOffset) * transferScale);
              col.a *= density;
              col.rgb *= col.a;
              accum += col * (1.0 - accum.a);
              if (accum.a > 0.99f)
                break;
            }
            return accum * brightness;
          }
        ''')
        self.updateShaderParams()
        
    def render(self):
        if self.volumeTex is None:
            return
        with ctx(self.traceVP, self.traceFP):
            drawQuad()
        
        
if __name__ == "__main__":    

    class App(ZglAppWX):
        volumeRender = Instance(VolumeRenderer)    
        
        def __init__(self):
            ZglAppWX.__init__(self, viewControl = FlyCamera())

            data = fromfile("img/bonsai.raw", uint8)
            data.shape = (256, 256, 256)
            #data = random.rand(256, 256, 256).astype(float32)
            #data[100:150,100:150] = 0

            data = swapaxes(data, 0, 1)
            self.volumeRender = VolumeRenderer(Texture3D(img=data))

        def display(self):
            clearGLBuffers()
            with ctx(self.viewControl.with_vp):
                self.volumeRender.render()


    if __name__ == "__main__":
        App().run()
