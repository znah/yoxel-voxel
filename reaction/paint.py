# -*- coding: utf-8 -*-
from zgl import *
    
def norm(a):
    return sqrt( sum(a*a, -1) )

class App(ZglAppWX):
    brush_radius = Range(5, 100, 30)
    blend_coef   = Range(0.0, 1.0, 0.1)
    
    def __init__(self):      
        self.grid_size  = grid_size = V(800, 600)
        ZglAppWX.__init__(self, size = grid_size, viewControl = OrthoCamera())

        self.flow_map = RenderTexture(size = grid_size / 2)
        self.flow_map.tex.filterLinear()
        
        def reset_flow():
            with self.flow_map:
                clearGLBuffers((0.5, 0.5, 0.5, 0.5))
        self.key_SPACE = reset_flow
        reset_flow()
        
        self.paintFP = genericFP('''
          // f2_pos, f2_r, f2_dir
          float2 dp = (tc0.xy - f2_pos) / f2_r;
          float d = length(dp);
          float v = smoothstep(1.0, 0.0, d);
          return float4(f2_dir, 1, v*0.1);
        ''')

        self.convect_buf = convect_buf = PingPong(size = grid_size)
        convect_buf.texparams(*Texture2D.Linear)

        self.convectFP = CGShader('fp40', '''
          uniform sampler2D flow;
          uniform sampler2D src;
          uniform sampler2D noise;
          uniform float2 gridSize;
          uniform float time;
          uniform float blend_coef;

          float get_noise(float2 p)
          {
            const float pi = 3.141593;
            float4 rnd = tex2D(noise, p);
            float v = rnd.r;
            float f = 1.0 + (rnd.g-0.5);
            //return v;
            //return frac(v+time);
            return 0.5*sin(2*pi*v+f*time*5)+0.5;
          }

          float4 main(float2 p : TEXCOORD0) : COLOR
          {
            float2 v = 2.0*tex2D(flow, p).xy-float2(1.0);
            float vel = length(v);
            float2 sp = p-v/gridSize;
            float4 c = tex2D(src, sp);
            float r = get_noise(p);
            c = lerp(c, float4(r), blend_coef);
            c = lerp(float4(0.5), c, 1.0+vel*0.2);
            return c;
          }
        ''')

        self.visFP = genericFP('''
          // s_flow, s_dye
          float2 p = tc0.xy;
          float4 flow = tex2D(s_flow, p);
          float4 dye  = tex2D(s_dye, p);
          float2 v = 2.0*flow.xy-float2(1.0);

          float vel = length(v);
          return dye*saturate(vel*1.5);
        ''')

        a = uint8(random.rand(grid_size[1], grid_size[0], 4) * 255)
        self.noise_tex = Texture2D(img=a)

        def update_vis():
            self.convectFP(
              flow = self.flow_map.tex,
              src = convect_buf.src.tex,
              noise = self.noise_tex,
              gridSize = grid_size,
              time = self.time,
              blend_coef = self.blend_coef)
              
              
            with ctx(self.convect_buf.dst, ortho01, self.convectFP):
                drawQuad()
            self.convect_buf.flip()
        self.update_vis = update_vis
        

    def OnMouse(self, evt):
        if evt.LeftIsDown():
            view_sz = self.viewControl.vp.size
            scr_pos = V(evt.Position)
            pos = scr_pos / view_sz
            pos[1] = 1.0 - pos[1]

            r = self.brush_radius / self.grid_size
            rect = V(pos - r, pos + r)
            dp = scr_pos - self.viewControl.mPos
            dp[1] *= -1
            d = norm(dp)
            if d > 0:
                dp = 0.5*dp/d + 0.5
                self.paintFP(
                  f2_pos = pos,
                  f2_r   = r,
                  f2_dir = dp
                )
                with ctx(self.flow_map, ortho01, glstate(GL_BLEND), self.paintFP):
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                    glBlendEquation(GL_FUNC_ADD);
                    draw_rect(rect, tex_rect=rect)
        if evt.WheelRotation != 0:
            wheel = int(evt.WheelRotation / evt.WheelDelta)
            self.brush_radius += wheel * 10
            

        ZglAppWX.OnMouse(self, evt)




    def display(self):
        clearGLBuffers()
        self.update_vis()
        self.visFP(
          s_flow = self.flow_map.tex,
          s_dye  = self.convect_buf.src.tex
        )
        with ctx(self.viewControl.vp, ortho01, self.visFP):
            drawQuad()

if __name__ == "__main__":
    App().run()
