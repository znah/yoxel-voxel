from __future__ import with_statement
from zgl import *
import wx
from wx import glcanvas

class ZglWXWindow(object):
    def __init__(self, title = "ZglWXApp", size = (800, 600)):
        self.client = None
        self.frame = frame = wx.Frame(None, wx.ID_ANY, title)
        frame.SetClientSize(size)
        self.canvas = canvas = glcanvas.GLCanvas(frame, -1)
        canvas.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
        canvas.Bind(wx.EVT_SIZE, self.OnSize)
        canvas.Bind(wx.EVT_PAINT, self.OnPaint)
        canvas.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        canvas.Bind(wx.EVT_KEY_UP, self.OnKeyUp)

        canvas.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)
        canvas.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)
        canvas.Bind(wx.EVT_MOTION, self.OnMouseMotion)

        canvas.Bind(wx.EVT_IDLE, self.OnMouseMotion)
       
        frame.Show(True)
        canvas.SetCurrent()

    def OnEraseBackground(self, event):
        pass # Do nothing, to avoid flashing on MSW.

    def OnPaint(self, event):
        dc = wx.PaintDC(self.canvas)
        safe_call(self.client, "display")
        self.canvas.SwapBuffers()

    def OnKeyDown(self, event):
        keycode = event.GetKeyCode()
        if keycode == wx.WXK_ESCAPE:
            self.frame.Close(True)

    def OnKeyUp(self, event):
        pass

    def OnSize(self, event):
        size = self.canvas.GetClientSize()
        safe_call(self.client, "resize", size.width, size.height)
        print size

    def OnMouseDown(self, event):
        pass
    def OnMouseUp(self, event):
        pass
    def OnMouseMotion(self, event):
        pass

def zglInitWX(viewSize, title):
    global wxApp
    wxApp = wx.PySimpleApp()
    wnd = ZglWXWindow(title = title, size = viewSize)
    InitCG()
    wxApp.SetTopWindow(wnd.frame)
    return wnd

def zglMainLoop():
    wxApp.MainLoop()



class App(ZglApp):
    def __init__(self):
        ZglApp.__init__(self, OrthoCamera())
        self.fragProg = CGShader('fp40', TestShaders, entry = 'TexCoordFP')
    
    def display(self):
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        with ctx(self.viewControl.with_vp, self.fragProg):
            drawQuad()

if __name__ == "__main__":
    wnd = zglInitWX((800, 600), "wx test")
    wnd.client = App()
    zglMainLoop()