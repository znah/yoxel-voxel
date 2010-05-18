from __future__ import with_statement
from zgl import *


class VolumeRenderer(HasTraits):
    density        = Range( 0.0, 1.0, 0.5)
    brightness     = Range( 0.0, 2.0, 1.0 )
    transferOffset = Range(-1.0, 1.0, 0.0 )
    transferScale  = Range(-50.0, 50.0, 1.0 )
    stepsInTexel   = Range( 0.5, 4.0, 2.0 )

    transferFunc   = Enum('nv', 'gray')

    sliceX         = Range(-1.0, 1.0, 0.0)
    sliceY         = Range(-1.0, 1.0, 0.0)
    sliceZ         = Range(-1.0, 1.0, 0.0)

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
            self.volumeTex.setParams(*Texture.ClampToEdge)
            self.traceFP.volume = self.volumeTex
            
    
    def __init__(self, volumeTex = None):
        HasTraits.__init__(self)
        
        self.volumeTex = volumeTex
            
        def makeTransTex(a):
            tex = Texture1D(a, format = GL_RGBA8)
            tex.filterLinear()
            tex.setParams(*Texture.ClampToEdge)
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
          #line 68

          float4 main(float2 p : POSITION, 
             out float3 rayDir : TEXCOORD0,
             out float3 eyePos : TEXCOORD1,
             out float2 screenPos : TEXCOORD2,
             out float3 eyeRayDir : TEXCOORD3) : POSITION
          {
            float4 projVec = float4(p * 2.0 - float2(1.0, 1.0), 0.0, 1.0);
            
            float4 wldVec = mul(glstate.matrix.inverse.mvp, projVec);
            wldVec /= wldVec.w;
            float4 eyeVec = mul(glstate.matrix.inverse.projection, projVec);
            eyeVec /= eyeVec.w;
            eyeRayDir = eyeVec.xyz;

            eyePos = mul(glstate.matrix.inverse.modelview[0], float4(0, 0, 0, 1)).xyz;
            rayDir = wldVec.xyz - eyePos;
            screenPos = p;
            return projVec;
          }
        
        ''')

        self.traceFP = CGShader('gp4fp', '''
        # line 93
          uniform sampler1D transferTex;
          uniform sampler3D volume;
          
          uniform float dt;
          uniform float density;
          uniform float brightness;
          uniform float transferOffset;
          uniform float transferScale;

          uniform float3 slice;

          uniform float useDepth;
          uniform sampler2D depthTex;
         
          void hitBox(float3 o, float3 d, out float tenter, out float texit)
          {
            float3 invD = 1.0 / d;
            float3 tlo = invD*(saturate(slice) - o);
            float3 thi = invD*(saturate(slice+1.0) - o);

            float3 t1 = min(tlo, thi);
            float3 t2 = max(tlo, thi);

            tenter = max(t1.x, max(t1.y, t1.z));
            texit  = min(t2.x, min(t2.y, t2.z));
          }

          float4 main( float3 rayDir    : TEXCOORD0, 
                       float3 eyePos    : TEXCOORD1,  
                       float2 screenPos : TEXCOORD2,
                       float3 eyeRayDir : TEXCOORD3) : COLOR 
          {
            rayDir = normalize(rayDir);
            float t1, t2;
            hitBox(eyePos, rayDir, t1, t2);
            t1 = max(0.0, t1);
            if (t1 > t2)
              return float4(0, 0, 0, 0);
            float3 p = eyePos + rayDir * t1;

            if (useDepth > 0.5)
            {
              float depth = tex2D(depthTex, screenPos);
              float4 v = float4( 2.0*float3(screenPos, depth)-1.0, 1.0 );
              v = mul( glstate.matrix.inverse.projection, v );
              v /= v.w;
              float rayZ = normalize(eyeRayDir).z;
              float t = v.z / rayZ;
              t2 = min(t2, t);
            }

            float step = dt;
            float4 accum = float4(0);
            for (float t = t1; t < t2; t += step, p += step * rayDir)
            {
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
        
    def render(self, depthTex = None):
        if self.volumeTex is None:
            return
        self.traceFP.slice = (self.sliceX, self.sliceY, self.sliceZ)
        if depthTex is not None:
            self.traceFP.depthTex = depthTex
            self.traceFP.useDepth = 1.0
        else:
            self.traceFP.useDepth = 0.0
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

            verts, idxs, quads = create_box()
            verts = 0.5 * verts + 0.25
            def drawBox():
                glColor(0.5, 0.5, 0.5)
                drawArrays(GL_QUADS, verts = verts, indices = quads)
            self.drawBox = drawBox

            self.depthTex = Texture2D()

            self.viewControl.zNear = 0.01
            self.viewControl.zFar  = 10.0

        def display(self):
            clearGLBuffers()
            with self.viewControl.with_vp:
                with glstate(GL_DEPTH_TEST):
                    self.drawBox()
                with ctx(self.depthTex):
                    w, h = self.viewSize
                    glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, 0, 0, w, h, 0)
                with glstate(GL_BLEND):
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                    glBlendEquation(GL_FUNC_ADD);
                    self.volumeRender.render(self.depthTex)

    if __name__ == "__main__":
        App().run()
