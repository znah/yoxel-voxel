from numpy import *

# 31 bit 
# - grid(0)
# - brick(1)
# special:
# 0xFF FF FF F0 - all zero
# 0xFF FF FF F1 - all one
ZeroBlock    = uint32(0xFFFFFFF0)
FullBlock    = uint32(0xFFFFFFF1)
BrickRefMask = uint32(0x80000000)

BrickSize = 4
GridSize = 4