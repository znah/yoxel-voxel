import numpy as np
import pyglet
import pyglet.gl as gl
from zgl2 import OrthoView, OverlayView, draw_arrays

class CrowdApp(pyglet.window.Window):
    def __init__(self):
        config = pyglet.gl.Config()#(sample_buffers=1, samples=4)
        super(CrowdApp, self).__init__(vsync=False, resizable=True, config=config)

        self.view = OrthoView(self)
        self.label = pyglet.text.Label('Hello, world!')
        self.fps_display = pyglet.clock.ClockDisplay()

        self.verts = np.float32( np.random.normal(size=(1e6, 2)) ).cumsum(0)

    def on_draw(self):
        self.clear()
        
        with self.view:
            draw_arrays(gl.GL_LINE_STRIP, self.verts)
            
        with OverlayView(self):
            self.fps_display.draw()


if __name__ == '__main__':
    window = CrowdApp()
    pyglet.app.run()
