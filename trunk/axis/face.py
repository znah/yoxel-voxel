import cv

cascade_fn = 'data/haarcascade_frontalface_alt2.xml'
cascade = cv.Load(cascade_fn)

def detect(img):
    global cascade

    min_size = (20, 20)
    image_scale = 2
    haar_scale = 1.2
    min_neighbors = 3
    haar_flags = 0

    w, h = cv.GetSize(img)
    gray = cv.CreateMat(h, w, cv.CV_8U)
    small = cv.CreateMat(h/image_scale, w/image_scale, cv.CV_8U)
    cv.CvtColor(img, gray, cv.CV_BGR2GRAY)
    cv.Resize(gray, small)
    
    
    #t = cv.GetTickCount()
    faces = cv.HaarDetectObjects(small, cascade, cv.CreateMemStorage(0),
                                 haar_scale, min_neighbors, haar_flags, min_size)
    #t = cv.GetTickCount() - t
    #print "detection time = %gms" % (t/(cv.GetTickFrequency()*1000.))

    sc = image_scale
    return [(x*sc, y*sc, w*sc, h*sc) for (x, y, w, h), n in faces]


if __name__ == '__main__':
    cap = cv.CreateCameraCapture(0)

    frame = cv.QueryFrame(cap)
    while True:
        faces = detect(frame)
        vis = cv.CloneImage(frame)
        for x, y, w, h in faces:
            cv.Rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv.ShowImage('face', vis)
        ch = cv.WaitKey(10)
        if ch == 27:
            break
        frame = cv.QueryFrame(cap)

        