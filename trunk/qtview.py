#!/usr/bin/env python

import sys
from PyQt4 import QtCore, QtGui

import pycuda.driver as cuda
from numpy import *
from time import clock

from ore.ore import *

import trace_cuda as voxel

class Window(QtGui.QWidget):
    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self, parent)

        self.pos = array([0.5, 0.5, 0.5])
        self.hang = -135.0
        self.vang = -55.0
        self.calcDir()

        self.renderer = voxel.CudaRenderer( (640, 480) )
        viewSize = self.renderer.getViewSize()

        self.scene = DynamicSVO()
        self.scene.Load("data/scene.vox")
        self.cudaScene = CudaSVO()
        self.cudaScene.SetSVO(self.scene)
        self.renderer.updateScene(self.cudaScene)

        self.sphereSrc = MakeShpereSource(64, (128, 128, 192), False)
        self.invSphereSrc = MakeShpereSource(64, (192, 128, 128), True)
         
        self.setFixedSize(viewSize[0], viewSize[1])
        self.setWindowTitle(self.tr("Interactive voxel"))

        self.moveFwd = 0
        self.moveSide = 0
        self.lastTime = clock()

        self.lastEditTime = clock()
        self.editState = 0

        timer = QtCore.QTimer(self)
        self.connect(timer, QtCore.SIGNAL("timeout()"), self.updatePos)
        timer.start(0)

    def mousePressEvent(self, event):
        self.lastPos = QtCore.QPoint(event.pos())

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & QtCore.Qt.LeftButton:
            self.hang -= dx*0.2
            self.vang = clip(self.vang-dy*0.2, -90, 90)
            self.calcDir()
            self.update()
        
        self.lastPos = QtCore.QPoint(event.pos())


    def getTargetPoint(self):
        t = self.scene.TraceRay( p3f(self.pos), p3f(self.viewDir) )
        return self.pos + self.viewDir * t

    def updateScene(self):
        start = clock()
        self.cudaScene.Update()
        self.renderer.updateScene(self.cudaScene)
        dt = clock() - start
        print "update time: %f ms" % (dt*1000)
        print self.scene.CountChangedPages(), "pages changed",
        print "%.2f MB to transfer" % (self.scene.CountTransfrerSize()/1024.0**2)


    def keyPressEvent(self, event):
        key = event.key()

        if key == QtCore.Qt.Key_Escape:
            self.close()
        elif key == QtCore.Qt.Key_W:
            self.moveFwd = 1
        elif key == QtCore.Qt.Key_S:
            self.moveFwd = -1
        elif key == QtCore.Qt.Key_D:
            self.moveSide = 1
        elif key == QtCore.Qt.Key_A:
            self.moveSide = -1

        elif key == QtCore.Qt.Key_9:
            self.renderer.detailCoef -= 0.2
        elif key == QtCore.Qt.Key_0:
            self.renderer.detailCoef += 0.2

        elif key == QtCore.Qt.Key_1:
            self.editState = 1
        elif key == QtCore.Qt.Key_2:
            self.editState = 2
            
        else:
            QtGui.QWidget.keyPressEvent(self, event)

    def keyReleaseEvent(self, event):
        key = event.key()

        if key == QtCore.Qt.Key_W:
            self.moveFwd = 0
        elif key == QtCore.Qt.Key_S:
            self.moveFwd = 0
        elif key == QtCore.Qt.Key_D:
            self.moveSide = 0
        elif key == QtCore.Qt.Key_A:
            self.moveSide = 0

        elif key == QtCore.Qt.Key_1:
            self.editState = 0
        elif key == QtCore.Qt.Key_2:
            self.editState = 0
        
        else:
            QtGui.QWidget.keyReleaseEvent(self, event)


    def updatePos(self):
        t = clock()
        dt = t - self.lastTime
        self.lastTime = t
        self.pos += dt*self.viewDir*self.moveFwd * 0.25

        a = (self.hang-90.0)/180.0*pi
        sdir = array([cos(a), sin(a), 0])
        self.pos += dt*sdir*self.moveSide * 0.25

        #self.pos = clip(self.pos, 0.0, 0.99999)

        lt = t*2
        #lightPos = (0.25 + 0.25*cos(lt), 0.25 + 0.25*sin(lt), 0.5)
        lightPos = self.pos
        self.renderer.setLightPos(lightPos)

        if (self.editState != 0):
            dstLevel = 11
            pos = self.getTargetPoint() * 2**dstLevel

            if self.editState == 1:
                self.scene.BuildRange(dstLevel, p3i(pos), BuildMode.GROW, self.sphereSrc) 
            else:
                self.scene.BuildRange(dstLevel, p3i(pos), BuildMode.CLEAR, self.invSphereSrc) 

            self.updateScene()    
            self.lastEditTime = t

        self.update()
        

    def calcDir(self):
        def d2r(a):
            return a/180.0*pi
        h = d2r(self.hang)
        v = d2r(self.vang)
        cv = cos(v)
        vdir = array( (cos(h)*cv, sin(h)*cv, sin(v)) )
        norm = sqrt( (vdir*vdir).sum() )
        self.viewDir = vdir / norm

    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)

        try:
            stat = self.renderer.render(self.pos, self.viewDir)
        except:
            self.close()

        img = flipud(self.renderer.getImage()).copy()
        img[img.shape[0]/2, img.shape[1]/2] = [255, 0, 0]

        qimg = QtGui.QImage(img, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        painter.drawImage(0, 0, qimg)

        painter.setPen(QtCore.Qt.white)
        painter.drawText(0, 0, 200, 200,0, stat)

        painter.end()





if __name__ == "__main__":
    cuda.init()
    assert cuda.Device.count() >= 1
    dev = cuda.Device(0)
    ctx = dev.make_context()

    raw_input("press enter to go")


    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
