from numpy import *

def saveImage(buf, fname):
    from PIL import Image
    im = Image.fromarray(flipud(buf))
    im.save(fname)

def renderTurn(renderer, frame_num=36, fn_pattern="_out/frame%02d.png"):
    from time import clock

    t_sum = 0.0
    for i in xrange(frame_num):
        ang = 2*pi/frame_num * i
        vdir = (cos(ang), sin(ang), -0.5)

        t = clock()
        renderer.render(vdir, i==0)
        dt = clock() - t
        if i > 0:
            t_sum += dt

        saveImage(renderer.getImage(), fn_pattern % i)
        print "frame %d saved - %f ms" % (i, dt*1000)
    
    print "avg time = ", t_sum/(frame_num-1)*1000
