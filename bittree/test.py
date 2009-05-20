from numpy import *
from pylab import *

def process(a, level):
    '''
    Returns (b, h), where 'h' marks voxels on isosurface and 'b' 
    is a copy of 'a' with internal samples set to 255 and external to 0.
    '''
    def corn(a):
        a1 = a[1:] | a[:-1]
        a2 = a1[:,1:] | a1[:,:-1]
        a3 = a2[:,:,1:] | a2[:,:,:-1]
        return a3

    def mark(a):
        res = zeros(array(a.shape) + 1, a.dtype)
        res[:-1, :-1, :-1] |= a
        res[:-1, :-1, 1: ] |= a
        res[:-1, 1:, :-1] |= a
        res[:-1, 1:, 1: ] |= a
        res[1:, :-1, :-1] |= a
        res[1:, :-1, 1: ] |= a
        res[1:, 1:, :-1] |= a
        res[1:, 1:, 1: ] |= a
        return res
    
    lo = a <= level
    hi = a > level
    h = corn(lo) & corn(hi)
    m = mark(h)
    m_in = ~m & hi
    m_out = ~m & lo
    b = a.copy()
    b[m_in] = 255
    b[m_out] = 0
    return (b, h)


def pad_bricks(a, bricksize):
    bs = int(bricksize)
    sza = array(a.shape)
    szb = (sza + (bs - 1)) / bs * bs
    b = zeros(szb, a.dtype)
    b[:sza[0], :sza[1], :sza[2]] = a
    return b


def reduce_bricks(a, bricksize, func, init):
    bs = int(bricksize)
    sz = array(a.shape) / bs
    res = array(sz, a.dtype)
    res[:] = init

    idx = [(i, j, k) for i in xrange(bs) for j in xrange(bs) for k in xrange(bs)]
    for (i, j, k) in idx:
        res = func(res, a[i::bs, j::bs, k::bs])
    return res

def make_bits(a):
    bs = 4
    sz = array(a.shape) / bs
    bits = zeros(sz, uint64)
    idx = [(i, j, k) for i in xrange(bs) for j in xrange(bs) for k in xrange(bs)]
    for (bit, (i, j, k)) in enumerate(idx):
        sub_a = a[i::bs, j::bs, k::bs].astype(uint64)
        bits |= (sub_a & 1) << bit
    return bits

def build_hinttree(a, gridsize = 4):
    a = pad_bricks(a, 4)
    bits = make_bits(a)
    bits = pad_bricks(gridsize)

    






if __name__ == '__main__':
    a = fromfile("../data/bonsai.raw", uint8)
    a.shape = (256, 256, 256)

    (b, h) = process(a, 32)
    h = pad_bricks(h, 4)
    #b.tofile("../data/bonsai32.raw")



