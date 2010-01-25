from __future__ import with_statement
from zgl import *
    
class App(ZglAppWX):
    def __init__(self):
        ZglAppWX.__init__(self, viewControl = OrthoCamera())
        self.flowProg = CGShader('fp40', '''
          uniform sampler2D noiseTex;
          uniform sampler2D srcTex;
          uniform float2 seed;
          uniform float blendCoef;

          float4 main(float2 pos : TEXCOORD0) : COLOR
          {
            float4 c1 = tex2D(srcTex, pos - float2(1, 0.7)*0.002);
            float4 c2 = tex2D(noiseTex, pos + seed);
            return lerp(c1, c2, 0.5);
          }
        ''')

        self.lookupProg = CGShader('fp40', TestShaders, entry = 'TexLookupFP')

        a = random.rand(512, 512, 4).astype(float32)
        a[a < 0.9] = 0
        noiseTex = Texture2D(a, format=GL_RGBA8)
        self.flowProg.noiseTex = noiseTex
        self.blendCoef = 0.2

        self.pp = PingPong(size = (512, 512))
        self.pp.texparams(*Texture2D.Linear)
    
    def display(self):
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.flowProg.srcTex = self.pp.src.tex
        self.flowProg.seed = random.rand(2)
        with ctx(self.pp.dst, ortho, self.flowProg):
            drawQuad()
        self.pp.flip()
        
        self.lookupProg.tex = self.pp.src.tex
        with ctx(self.viewControl.with_vp, self.lookupProg):
            drawQuad()

if __name__ == "__main__":
    App().run()
