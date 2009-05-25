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

def make_bits(a):
    bs = 4
    sz = array(a.shape) / bs
    bits = zeros(sz, uint64)
    idx = ndindex(bs, bs, bs)
    for (bit, (i, j, k)) in enumerate(idx):
        sub_a = a[i::bs, j::bs, k::bs].astype(uint64)
        bits |= (sub_a & 1) << bit
    return bits


# 31 bit 
# - grid(0)
# - brick(1)
# special:
# 0xFF FF FF F0 - all zero
# 0xFF FF FF F1 - all one
ZeroBlock    = uint32(0xFFFFFFF0)
FullBlock    = uint32(0xFFFFFFF1)
BrickRefMask = uint32(0x80000000)

def build_level(prevLevel, gridsize, base_ofs):
    gs = gridsize
    gs3 = gs ** 3
    prevLevel = pad_bricks(prevLevel, gs)
    newLevelSize = array(prevLevel.shape) / gs
    grids = zeros(tuple(newLevelSize) + (gs3,), uint32)
    for (ofs, (i, j, k)) in enumerate(ndindex(gs, gs, gs)):
        grids[:,:,:,ofs] = prevLevel[i::gs, j::gs, k::gs]

    zero_refs = grids == ZeroBlock
    full_refs = grids == FullBlock
    zero_grids = zero_refs.all(3)
    full_grids = full_refs.all(3)
    grid_flags = ~(zero_grids | full_grids)
    
    packed_level = grids[grid_flags].copy()
    nextLevel = grid_flags.astype(int32).cumsum()-1 + base_ofs
    nextLevel.shape = grid_flags.shape
    nextLevel[zero_grids] = ZeroBlock
    nextLevel[full_grids] = FullBlock
    return (packed_level, nextLevel)

def build_bittree(a, gridsize = 4):
    a = pad_bricks(a, 4)
    bits = make_bits(a)
    bits = pad_bricks(bits, gridsize)

    zero_flags = bits == 0
    full_flags = bits == ~uint64(0)
    brick_flags = ~(zero_flags | full_flags)
    packed_bricks = bits[brick_flags].copy()

    refs = brick_flags.astype(int32).cumsum()-1
    refs.shape = bits.shape
    refs |= BrickRefMask
    refs[zero_flags] = ZeroBlock
    refs[full_flags] = FullBlock

    levels = []
    base_ofs = 0
    while refs.size > 1:
        print refs.shape
        (packed_level, refs) = build_level(refs, gridsize, base_ofs)
        base_ofs += len(packed_level)
        levels.append(packed_level)
    packed_grids = vstack(levels)
    return (packed_grids, packed_bricks)



if __name__ == '__main__':
    print "loading data"
    a = fromfile("../data/bonsai.raw", uint8)
    a.shape = (256, 256, 256)

    print "marking voxels"
    (b, h) = process(a, 32)
    print "building tree"
    (grids, bit_bricks) = build_bittree(h, 4)
    print "grid num: %d  brick num: %d" % (len(grids), len(bit_bricks))

    print "saving result"
    grids.tofile("grids.dat")
    bit_bricks.tofile("bit_bricks.dat")



