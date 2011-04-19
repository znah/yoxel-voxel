import urllib2
import cv
from time import sleep

cameras = ['192.168.255.%d' % i for i in xrange(100, 104)]

restart_url = "http://%s/axis-cgi/mpeg4/restart_stream.cgi"
ptz_url = "http://%s/axis-cgi/com/ptz.cgi"


def setup_auth(cameras, user, passwd):
    passman = urllib2.HTTPPasswordMgrWithDefaultRealm()
    for cam in cameras:
        passman.add_password(None, 'http://%s/' % cam, user, passwd)
    authhandler = urllib2.HTTPBasicAuthHandler(passman)
    opener = urllib2.build_opener(authhandler)
    urllib2.install_opener(opener)

setup_auth(cameras, 'root', 'root')

def cam_ptz_command(cam, cmd, *args):
    url = (ptz_url % cam) + ("?%s=" % cmd) + ','.join(map(str, args))
    f = urllib2.urlopen(url)
    return f.read()

def cam_mpeg_restart(cam):
    url = restart_url % cam
    f = urllib2.urlopen(url)

def parse_keyval(lines):
    d = {}
    for s in lines.split():
        key, val = s.split('=')
        try:
            val = float(val)
        except ValueError:
            val = dict(on=True, off=False).get(val, val)
        d[key] = val
    return d

def get_cam_ptz(cam):
    res = cam_ptz_command(cam, 'query', 'position')
    return parse_keyval(res)

if __name__ == '__main__':
    '''
    for cam in cameras:
        cam_ptz_command(cam, 'pan', 0)
        cam_ptz_command(cam, 'tilt', 0)
        print cam
    '''
    cam = cameras[0]

    cam_ptz_command(cam, 'tilt', -20)

    speed = 10
    cam_ptz_command(cam, 'continuouspantiltmove', speed, 0)

    while True:
        pan = get_cam_ptz(cam)['pan']
        if pan > 30:
            cam_ptz_command(cam, 'continuouspantiltmove', -speed,0)
            print pan
        if pan < -30:
            cam_ptz_command(cam, 'continuouspantiltmove', speed, 0)
            print pan
        sleep(0.1)
