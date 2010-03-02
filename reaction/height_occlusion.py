from __future__ import with_statement
from zgl import *
    
    
class Blur:
    def __init__(self, size):
        self.pp = PingPong(size = size, format = GL_LUMINANCE_FLOAT32_ATI)
        self.prog = CGShader('fp40', '''
          uniform sampler2D src;
          uniform float2 d;
          
          float4 main(float2 p : TEXCOORD0) : COLOR
          {
            return (tex2D(src, p-d) + tex2D(src, p) + tex2D(src, p+d)) / 3.0;
          }        
        ''')
        self.dx = (1.0 / size[0], 0)
        self.dy = (0, 1.0 / size[1])
        
    def __call__(self, src, n):
        if n == 0:
            return src
        self._pass(src, self.pp.src, self.dx)
        self._pass(self.pp.src.tex, self.pp.dst, self.dy)
        self.pp.flip()
        for i in xrange(n-1):
            self._pass(self.pp.src.tex, self.pp.dst, self.dx)
            self.pp.flip()
            self._pass(self.pp.src.tex, self.pp.dst, self.dy)
            self.pp.flip()
        return self.pp.src.tex
        
    def _pass(self, src, dst, d):
        with ctx(dst, ortho, self.prog(src = src, d = d)):
            drawQuad()
    
    
    
class App(ZglAppWX):
    def __init__(self):
        ZglAppWX.__init__(self, viewControl = FlyCamera())
        self.viewControl.speed = 1000.0
        self.viewControl.zNear = 100.0
        self.viewControl.zFar = 30000.0

        a = fromfile("data/h2049.dat", float32).reshape(2049, 2049)
        self.heightmap = heightmap = Texture2D(img=a, format = GL_LUMINANCE_FLOAT32_ATI)
        self.occlusion = RenderTexture(size = heightmap.size)

        self.vertProg = CGShader("vp40", """
          uniform sampler2D heightmap : TEXUNIT1;
          uniform float2 terrainSize;

          void main(
            float2 pos : ATTR0,
            out float4 oPos: POSITION,
            out float2 oTC: TEXCOORD0)
          {
            float h = tex2D(heightmap, pos).r;
            float4 wldPos = float4(pos * terrainSize, h, 1.0);
            oPos = mul(glstate.matrix.mvp, wldPos);
            oTC = pos;
          }
        """)
        self.vertProg.heightmap = heightmap
        self.vertProg.terrainSize = heightmap.size * 10.0
        
        self.occlusion2Frag = CGShader('fp40', '''
          uniform sampler2D heightmap;
          uniform sampler2D blurred;
          uniform float coef;
          uniform float occlPow;
          
          float4 main(float2 p : TEXCOORD0) : COLOR
          {
            float h = tex2D(heightmap, p);
            float b = tex2D(blurred, p);
            float occl = saturate(1.0 - (b - h) * coef);
            return pow(occl, occlPow);
          }
        ''')
        self.occlusion2Frag.heightmap = heightmap
        
        self.texFrag = CGShader('fp40', TestShaders, entry = 'TexLookupFP')
        self.texFrag.tex = self.occlusion.tex
        self.occlusion.tex.filterLinear()
        
        self.blur = Blur(heightmap.size)

        self.updateOcclusion()        

    #maxSampleDist = Range(10.0, 300.0, 100.0)    
    #occlusionPower = Range(1.0, 16.0, 8.0)    
    #distFadePower = Range(0.1, 100.0, 2.0)    
    occlusionCoef = Float(0.015)
    occlusionPow = Float(1.0)
    blurIterNum = Int(100)
    afterBlur = Int(1)

    @on_trait_change( '+' )
    def updateOcclusion(self):
        blurred = self.blur(self.heightmap, self.blurIterNum)
        self.occlusion2Frag.blurred = blurred
        self.occlusion2Frag(coef = self.occlusionCoef, occlPow = self.occlusionPow)
        with ctx(self.occlusion, ortho, self.occlusion2Frag):
            drawQuad()
            
        self.texFrag.tex = self.blur(self.occlusion.tex, self.afterBlur)    
        self.texFrag.tex.filterLinear()
        
        
        '''
        self.occlusionFrag(
          maxSampleDist = self.maxSampleDist, 
          occlusionPower = self.occlusionPower,
          distFadePower = self.distFadePower)
        with ctx(self.occlusion, ortho, self.occlusionFrag):
            drawQuad()
        '''    


    def display(self):
        clearGLBuffers()
        self.vertProg.heightmap = self.heightmap
        with ctx(self.viewControl.with_vp, self.vertProg, self.texFrag, glstate(GL_DEPTH_TEST)):
            drawGrid(1025, 1025)

if __name__ == "__main__":
    App().run()
