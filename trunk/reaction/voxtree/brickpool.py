from __future__ import with_statement
import sys
sys.path.append('..')
from zgl import *
from cutools import *
import _brickpool as bp

class BrickPool(HasTraits):
    brick_size = ReadOnly()
    pool_shape = ReadOnly()

    map_slot_num = ReadOnly(16)

    d_pool_array    = ReadOnly()
    d_map_slots     = ReadOnly()
    d_map_mark_enum = ReadOnly()

    _ = Python(editable = False)
    
    def __init__(self, brick_size = 5, pool_shape = (64, 128, 64)):
        self.brick_size = brick_size
        self.pool_shape = pool_shape
        w = pool_shape[2] * brick_size
        h = pool_shape[1] * brick_size

        descr = setattrs(cu.ArrayDescriptor3D(),
            width  = w,
            height = h,
            depth  = pool_shape[0] * brick_size,
            format = cu.dtype_to_array_format(uint8),
            num_channels = 1,
            flags = 0 )
        self.d_pool_array = cu.Array(descr)

        self.d_map_slots = d_map_slots = ga.zeros((self.map_slot_num, brick_size, h, w), uint8) 
        self.d_map_mark_enum = ga.zeros((self.map_slot_num, pool_shape[1], pool_shape[2]), int32) 
        
        pool2map = setattrs( cu.Memcpy3D(), 
            width_in_bytes = w,
            height         = h,
            depth          = brick_size,
            
            src_x_in_bytes = 0,
            src_y          = 0,
            src_z          = 0, # to set
            src_lod        = 0,

            dst_pitch      = w,
            dst_height     = h,
            dst_x_in_bytes = 0,
            dst_y          = 0,
            dst_z          = 0, # to set
            dst_lod        = 0 )
        pool2map.set_src_array  (self.d_pool_array)
        pool2map.set_dst_device (d_map_slots.gpudata)

        map2pool = setattrs( cu.Memcpy3D(), 
            width_in_bytes = w,
            height         = h,
            depth          = brick_size,
            
            dst_x_in_bytes = 0,
            dst_y          = 0,
            dst_z          = 0, # to set
            dst_lod        = 0,

            src_pitch      = w,
            src_height     = h,
            src_x_in_bytes = 0,
            src_y          = 0,
            src_z          = 0, # to set
            src_lod        = 0 )
        map2pool.set_dst_array  (self.d_pool_array)
        map2pool.set_src_device (d_map_slots.gpudata)

        params = setattrs(bp.CuBrickPoolManager_Params(),
            sizeX = pool_shape[2],
            sizeY = pool_shape[1],
            sizeZ = pool_shape[0],
            mappingSlotNum     = self.map_slot_num,
            d_mapSlotsMarkEnum = int(self.d_map_mark_enum.gpudata) )
        pool_man = bp.CuBrickPoolManager(params)

        def allocMap(n):
            allocatedNum = pool_man.allocMap(n)
            self.slot2slice = pool_man.slot2slice
            for slot, poolSlice in enumerate(self.slot2slice):
               pool2map.src_z = int(poolSlice * brick_size)
               pool2map.dst_z = slot * brick_size
               pool2map()
            return allocatedNum
        self.allocMap = allocMap
            
        def commit():
            for slot, poolSlice in enumerate(self.slot2slice):
               map2pool.dst_z = int(poolSlice * brick_size)
               map2pool.src_z = slot * brick_size
               map2pool()
        self.commit = commit
        
