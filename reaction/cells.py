from __future__ import with_statement
from zgl import *

shaderCode = '''
  uniform sampler2D noiseTex;
  uniform float2 noiseSize;
  uniform float time;

  #define RADIUS 0.02
  #define CELL_SIZE RADIUS
  #define DENSITY 10000
  #define CELL_PROB (DENSITY * CELL_SIZE * CELL_SIZE)
  #define PI 3.141593

  float4 noise(inout float2 seed)
  {
    float4 v = tex2D(noiseTex, seed);
    seed = v.zw;
    return v;
  }

  float cell(float2 ci, float2 pos)
  {
    float2 seed = ci / noiseSize;
    float count = 0;
    float l = exp(-CELL_PROB), p = noise(seed).x;
    while (p > l)
    {
      count += 1;
      p *= noise(seed).x;
    }

    float sum = 0;
    for (float i = 0; i < count; i += 1)
    {
      float3 v = noise(seed).xyz;
      float2 center = v.xy;
      float r = v.z * abs(sin(time+seed.x*10));
      float d = length(center - pos) / r;
      if (d < 1)
        sum += sin(d*PI*3) * 0.3;
    }

    return sum;
  }

  float4 main(float2 pos: TEXCOORD0) : COLOR
  {
    float2 p2 = pos / CELL_SIZE;
    float2 ci = floor(p2);
    float2 posInCell = p2 - ci;
    float sum = 0;
    for (int yi = -1; yi <= 1; ++yi)
      for (int xi = -1; xi <= 1; ++xi)
      {
        float2 dp = float2(xi, yi);
        sum += cell(ci + dp, posInCell - dp);
      }

    return float4(sum, sum, sum, 1);

  }
'''

class App(ZglAppWX):
    def __init__(self):
        ZglAppWX.__init__(self, viewControl = OrthoCamera())

        self.cellFrag = CGShader("fp40", shaderCode)

        self.noiseTex = Texture2D(random.rand(512, 512, 4).astype(float32), format=GL_RGBA_FLOAT32_ATI)
        self.cellFrag.noiseTex = self.noiseTex
        self.cellFrag.noiseSize = self.noiseTex.size
    
    def display(self):
        clearBuffers()
        with ctx(self.viewControl.with_vp, self.cellFrag(time = self.time)):
            drawQuad()

if __name__ == "__main__":
    app = App()
    app.run()
