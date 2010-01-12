from __future__ import with_statement
from zgl import *
import wx
from wx import glcanvas

class ZglAppWX(object):
    def __init__(self, title = "ZglAppWX", size = (800, 600)):
        self.app = wx.PySimpleApp()
        self.frame = frame = wx.Frame(None, wx.ID_ANY, title)
        self.canvas = canvas = glcanvas.GLCanvas(frame, -1)
        self.initSize = size

        canvas.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
        canvas.Bind(wx.EVT_SIZE, self.OnSize)
        canvas.Bind(wx.EVT_PAINT, self.OnPaint)
        canvas.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        canvas.Bind(wx.EVT_KEY_UP, self.OnKeyUp)

        canvas.Bind(wx.EVT_MOUSE_EVENTS, self.OnMouse)

        #canvas.Bind(wx.EVT_IDLE, self.OnMouseMotion)
       
        frame.Show(True)
        canvas.SetCurrent()
        InitCG()

    def run(self):
        self.frame.SetClientSize(self.initSize)
        self.app.SetTopWindow(self.frame)
        self.app.MainLoop()


    def OnEraseBackground(self, event):
        pass # Do nothing, to avoid flashing on MSW.

    def OnPaint(self, event):
        dc = wx.PaintDC(self.canvas)
        safe_call(self, "display")
        self.canvas.SwapBuffers()

    def OnKeyDown(self, event):
        keycode = event.GetKeyCode()
        if keycode == wx.WXK_ESCAPE:
            self.frame.Close(True)

    def OnKeyUp(self, event):
        pass

    def OnSize(self, event):
        size = self.canvas.GetClientSize()
        #safe_call(self.client, "resize", size.width, size.height)
        self.canvas.Refresh(False)

    def OnMouse(self, evt):
        x, y = evt.Position
        if evt.IsButton():
            up = evt.ButtonUp()
            btn = evt.GetButton()
            print up, btn

class App(ZglAppWX):
    def __init__(self):
        ZglAppWX.__init__(self)
        self.viewControl = OrthoCamera()
        self.fragProg = CGShader('fp40', TestShaders, entry = 'TexCoordFP')
    
    def display(self):
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        with ctx(self.viewControl.with_vp, self.fragProg):
            drawQuad()

if __name__ == "__main__":
    app = App()
    app.run()