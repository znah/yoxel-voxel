import numpy as np
import cv
from numpy.ctypeslib import ndpointer
from ctypes import cdll, Structure, c_int, c_float, c_void_p
from ctypes.wintypes import BYTE, WORD, DWORD

dll_name = "CLEyeMulticam.dll"
dll = cdll.LoadLibrary(dll_name)

class GUID(Structure):
     _fields_ = [("Data1", DWORD),
             ("Data2", WORD),
             ("Data3", WORD),
             ("Data4", BYTE * 8)]

dll.CLEyeCreateCamera.argtypes = [c_int]  # camId
dll.CLEyeGetCameraUUID.restype = GUID

# camUUID, CLEyeCameraColorMode mode, CLEyeCameraResolution res, float frameRate
dll.CLEyeCreateCamera.argtypes = [GUID, c_int, c_int, c_float] 
dll.CLEyeCreateCamera.restype = c_void_p

# cam, ptr, timeout
dll.CLEyeCameraGetFrame.argtypes = [c_void_p, ndpointer(dtype=np.uint8, flags=('C_CONTIGUOUS', 'WRITEABLE')), c_int]


# camera modes (CLEyeCameraColorMode)
CLEYE_MONO_PROCESSED  = 0
CLEYE_COLOR_PROCESSED = 1
CLEYE_MONO_RAW        = 2
CLEYE_COLOR_RAW       = 3
CLEYE_BAYER_RAW       = 4

# camera resolution (CLEyeCameraResolution)
CLEYE_QVGA = 0
CLEYE_VGA  = 1

# camera parameters (CLEyeCameraParameter)
# camera sensor parameters
(CLEYE_AUTO_GAIN,            # [false, true]
CLEYE_GAIN,                 # [0, 79]
CLEYE_AUTO_EXPOSURE,        # [false, true]
CLEYE_EXPOSURE,             # [0, 511]
CLEYE_AUTO_WHITEBALANCE,    # [false, true]
CLEYE_WHITEBALANCE_RED,     # [0, 255]
CLEYE_WHITEBALANCE_GREEN,   # [0, 255]
CLEYE_WHITEBALANCE_BLUE,    # [0, 255]
# camera linear transform parameters (valid for CLEYE_MONO_PROCESSED, CLEYE_COLOR_PROCESSED modes)
CLEYE_HFLIP,                # [false, true]
CLEYE_VFLIP,                # [false, true]
CLEYE_HKEYSTONE,            # [-500, 500]
CLEYE_VKEYSTONE,            # [-500, 500]
CLEYE_XOFFSET,              # [-500, 500]
CLEYE_YOFFSET,              # [-500, 500]
CLEYE_ROTATION,             # [-500, 500]
CLEYE_ZOOM,                 # [-500, 500]
# camera non-linear transform parameters (valid for CLEYE_MONO_PROCESSED, CLEYE_COLOR_PROCESSED modes)
CLEYE_LENSCORRECTION1,      # [-500, 500]
CLEYE_LENSCORRECTION2,      # [-500, 500]
CLEYE_LENSCORRECTION3,      # [-500, 500]
CLEYE_LENSBRIGHTNESS        # [-500, 500]
) = range(20)


class CLEye(object):
    def __init__(self, fps = 60.0):
        self.guid = dll.CLEyeGetCameraUUID(0)
        self.cam = cam = dll.CLEyeCreateCamera( self.guid, CLEYE_COLOR_PROCESSED, CLEYE_VGA, fps )

        dll.CLEyeCameraStart(cam)
    
        dll.CLEyeSetCameraParameter(cam, CLEYE_AUTO_GAIN, 0)
        dll.CLEyeSetCameraParameter(cam, CLEYE_AUTO_EXPOSURE, 0)
        dll.CLEyeSetCameraParameter(cam, CLEYE_AUTO_WHITEBALANCE, 0)
        self.set_wb(255, 255, 255)
        self.gain = 20
        self.exposure = 128

        self.frame = np.zeros((480, 640, 4), np.uint8)

    def set_wb(self, r, g, b):
        dll.CLEyeSetCameraParameter(self.cam, CLEYE_WHITEBALANCE_RED, r)
        dll.CLEyeSetCameraParameter(self.cam, CLEYE_WHITEBALANCE_GREEN, g)
        dll.CLEyeSetCameraParameter(self.cam, CLEYE_WHITEBALANCE_BLUE, b)

    @property
    def gain(self):
        return dll.CLEyeGetCameraParameter(self.cam, CLEYE_GAIN)
    @gain.setter
    def gain(self, value):
        dll.CLEyeSetCameraParameter(self.cam, CLEYE_GAIN, value)
    def set_gain(self, value):
        self.gain = value

    @property
    def exposure(self):
        return dll.CLEyeGetCameraParameter(self.cam, CLEYE_EXPOSURE)
    @exposure.setter
    def exposure(self, value):
        dll.CLEyeSetCameraParameter(self.cam, CLEYE_EXPOSURE, value)
    def set_exposure(self, value):
        self.exposure = value

    def get_frame(self):
        dll.CLEyeCameraGetFrame(self.cam, self.frame, 2000) 
        return self.frame

    def add_sliders(self, window):
        cv.CreateTrackbar('gain', window, self.gain, 79, self.set_gain)
        cv.CreateTrackbar('exposure', window, self.exposure, 511, self.set_exposure)

    def stop(self):
        print 'stop!!!'
        dll.CLEyeCameraStop(self.cam)

        

if __name__ == '__main__':
    import cv, cv2

    print 'camera count: ', dll.CLEyeGetCameraCount()

    cam = CLEye(60.0)
    cv2.namedWindow('camera', 1)
    cam.add_sliders('camera')

    led = True
    while True:
        img = cam.get_frame()
        cv2.imshow( "camera", img )
            
        c = cv2.waitKey(1)
        if c == 27:
            break 
        if c == ord('1'):
            led = not led
            dll.CLEyeCameraLED(cam, led)

        if c == ord(' '):
            print 'capture...'
            rec_frames = np.zeros((60*5, 480, 640, 4), np.uint8)
            for dst in rec_frames:
                dst[:] = cam.get_frame()
                cv2.imshow( "camera", dst )
                cv2.waitKey(1)
            print 'capture done'
            
            def show_frame(i):
                cv2.imshow('rec', rec_frames[i])
            show_frame(0)
            cv.CreateTrackbar('frame', 'rec', 0, len(rec_frames)-1, show_frame)
    cam.stop()
