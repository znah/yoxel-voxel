from numpy import *

a = load("a_512.npy")

brick_size = bs = 8
border = 1
pbs = brick_size + border
grid_size = gs = array(a.shape) / brick_size

if border > 0:
    sz = array(a.shape)
    b = zeros(sz + border, uint8)
    b[:sz[0], :sz[1], :sz[2]] = a
    a = b


acc = zeros(grid_size, int32)
for i, j, k in ndindex(pbs, pbs, pbs):
    acc += a[i::bs, j::bs, k::bs] [:gs[0], :gs[1], :gs[2]]

SolidAcc = 255*pbs**3
mark = (acc > 0) & (acc < SolidAcc)


save('mark', mark.astype(uint8))


'''
print "..."

bz, by, bx = mgrid[ 0:gs[0], 0:gs[1], 0:gs[2] ]
bx = bx[mark]
by = by[mark]
bz = bz[mark]

bricknum = sum(mark)
bricks = zeros((bricknum, pbs, pbs, pbs), uint8)

bbx = bx * bs
bby = by * bs
bbz = bz * bs
for i, j, k in ndindex(pbs, pbs, pbs):
    bricks[:, i, j, k] = a[bbz+i, bby+j, bbx+k]

save("bricks", bricks)

print "..."

grid = zeros(grid_size, uint32)
grid[:] = 0x80000000
grid[acc == SolidAcc] = 0x80000001
grid[bz, by, bx] = arange(bricknum)
save("grid", grid)

print bricknum
print sum(mark) * pbs**3 / 2.0**20



'''