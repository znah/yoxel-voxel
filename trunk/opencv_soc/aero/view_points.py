# -*- coding: utf-8 -*-
from zgl import *
import cv2
    
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = hstack([verts, colors])
    verts = verts[verts[:,2] > verts[:,2].min()]
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        savetxt(f, verts, '%f %f %f %d %d %d')
    
    '''
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        for (x, y, z), (r, g, b) in zip(verts, colors):
           f.write( '%f %f %f %d %d %d\n' % (x, y, z, r, g, b) )
    '''    


class App(ZglAppWX):
    def __init__(self):
        ZglAppWX.__init__(self, viewControl = FlyCamera())
        fragProg = genericFP('tc0')

        src_idx = 0

        disp = load('%02d_disp.npy' % src_idx)
        h, w = disp.shape
        verts = zeros((h, w, 3), float32)
        verts[...,1], verts[...,0] = ogrid[:h,:w]
        verts[...,2] = disp
        verts[...,:] *= (0.1, 0.1, 0.5)
        verts *= 0.1

        '''
        idxgrid = arange(h*w).reshape(h, w)
        idxs = zeros((h-1, w-1, 4), uint32)
        idxs[...,0] = idxgrid[ :-1, :-1 ]
        idxs[...,1] = idxgrid[ :-1,1:   ]  
        idxs[...,2] = idxgrid[1:  ,1:   ]
        idxs[...,3] = idxgrid[1:  , :-1 ]
        idxs = idxs.flatten()
        '''

        colors = cv2.imread('%02d_l.bmp' % src_idx)[:,:,::-1]
        colors[disp == disp.min()] = 0
        colorsf = float32(colors) / 255.0

        write_ply('%02d_cloud.ply' % src_idx, verts, colors)
    
        def display():
            clearGLBuffers()
            with ctx(self.viewControl.with_vp, glstate(GL_DEPTH_TEST), fragProg):
                glPointSize(3.0)
                draw_arrays(GL_POINTS, verts=verts, tc0=colorsf)#, indices=idxs)
        self.display = display

if __name__ == "__main__":
    App().run()
