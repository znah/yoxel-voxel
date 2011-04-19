import cv
import axis
import face
 
import threading

from time import clock


cam_name = axis.cameras[0]

axis.cam_mpeg_restart(cam_name)

# rtsp://192.168.255.100/mpeg4/media.amp
cap = cv.CreateFileCapture( "rtsp://%s/mpeg4/media.amp" % cam_name)
#cap = cv.CreateCameraCapture( 0 )
w = cv.GetCaptureProperty(cap, cv.CV_CAP_PROP_FRAME_WIDTH)
h = cv.GetCaptureProperty(cap, cv.CV_CAP_PROP_FRAME_HEIGHT)
print w, h

cv.NamedWindow(cam_name)

work_thread = None

def async_cmd(cam, cmd, *args):
    global work_thread
    if work_thread is None or not work_thread.is_alive():
        def run():
            axis.cam_ptz_command(cam, cmd, *args)
        work_thread = threading.Thread(target=run)
        work_thread.start()

def onmouse(event, x, y, flags, param):
    if event != cv.CV_EVENT_LBUTTONDOWN:
        return
    async_cmd(cam_name, 'center', x, y)

def onzoom(value):
    async_cmd(cam_name, 'zoom', 1+int(value/100.0*15445))

def onfocus(value):
    async_cmd(cam_name, 'focus', 3637+int(value/100.0*(9999-3637)))


cv.SetMouseCallback(cam_name, onmouse)
cv.CreateTrackbar('zoom', cam_name, 0, 100, onzoom)
cv.CreateTrackbar('focus', cam_name, 0, 100, onfocus)

last_time = clock()
detect_rate = 1

def draw_faces(vis, faces):
    for x, y, w, h in faces:
        cv.Rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)

faces = []
shot_idx = 0

while True:
    frame = cv.QueryFrame(cap)
    vis = cv.CloneImage(frame)
    
    if clock() - last_time > detect_rate:
        last_time = clock()
        faces = face.detect(frame)
        draw_faces(vis, faces)
        if len(faces) > 0:
            cv.SaveImage('out/face_%03d.bmp' % shot_idx, vis)
            shot_idx += 1
            x, y, h, w = faces[0]
            async_cmd(cam_name, 'center', x + w/2, y + h/2)
    else:
        draw_faces(vis, faces)


    cv.ShowImage(cam_name, vis)
    ch = cv.WaitKey(10)
    if ch == 27:
        break
    if ch == ord('a'):
        print ch
        async_cmd(cam_name, 'rzoom', 2000)
    if ch == ord('z'):
        print ch
        async_cmd(cam_name, 'rzoom', -2000)

