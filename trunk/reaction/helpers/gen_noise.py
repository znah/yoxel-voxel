from numpy import *
from scipy import signal
from PIL import Image

def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = mgrid[-size:size+1, -sizey:sizey+1]
    g = exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

kern = gauss_kern(2)

def gen_noise(size):
    a = random.random(size)
    a = signal.convolve2d(a, kern, mode = 'same', boundary = 'wrap')
    a = (a - a.min()) / a.ptp()
    return a

size = (256, 256)
a = zeros(size + (4,))
for i in  xrange(4):
    a[..., i] = gen_noise(size)
a = (a*255).astype(uint8)

im = Image.fromarray(a, 'RGBA')
im.save("noise256x4g.png")
