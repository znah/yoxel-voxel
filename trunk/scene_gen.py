from numpy import *
from PIL import Image
from ore.ore import *
from scipy import weave
from scipy.weave import converters
from time import clock

def buildRegion(x0, y0, x1, y1, hmap):
    sx = slice( max(0, x0-1), min(hmap.shape[1], x1+1) )
    sy = slice( max(0, y0-1), min(hmap.shape[0], y1+1) )
    h0 = int(hmap[sy, sx].min())
    h1 = int(hmap[sy, sx].max())
    dh = h1-h0+1
    cmap = zeros((dh, y1-y0, x1-x0, 4), uint8)
    nmap = zeros((dh, y1-y0, x1-x0, 4), int8)
    code = '''
        #line 24 "scene_gen.py"
        using namespace cg;
        int sx = Nhmap[0], sy = Nhmap[1];
        for (int h = h0; h < (int)h0+(int)dh; ++h)
        for (int y = max(y0, 1); y < min(y1, sy-1); ++y)
        for (int x = max(x0, 1); x < min(x1, sx-1); ++x)
        {
            int ch = (int)hmap(y, x);
            if (h > ch)
                continue;

            int px = x-x0;
            int py = y-y0;
            int ph = h-(int)h0;
            
            if (h < ch)
            {
              bool flag = false;
              if ((int)hmap(y, x-1) < h)
                flag = true;
              if ((int)hmap(y-1, x) < h)
                flag = true;
              if ((int)hmap(y, x+1) < h)
                flag = true;
              if ((int)hmap(y+1, x) < h)
                flag = true;
              if (!flag)
              {
                cmap(ph, py, px, 3) = 1; // internal
                continue;
              }
            }
            float dx = (hmap(y, x+1) - hmap(y, x-1)) * 0.5f;
            float dy = (hmap(y+1, x) - hmap(y-1, x)) * 0.5f;
            point_3f n(-dx, -dy, 1);
            normalize(n);
            n *= 127.0;

            point_3f c1(0.1f, 0.5f, 0.1f), c2(0.7f, 0.3f, 0.2f);
            point_3b c = 255*(c1 + (h/300.0f) * (c2-c1));
            
            cmap(ph, py, px, 0) = c[0];
            cmap(ph, py, px, 1) = c[1];
            cmap(ph, py, px, 2) = c[2];
            cmap(ph, py, px, 3) = 255; // surface

            nmap(ph, py, px, 0) = n.x;
            nmap(ph, py, px, 1) = n.y;
            nmap(ph, py, px, 2) = n.z;
            nmap(ph, py, px, 3) = 0;
        }

    '''
    weave.inline(code, ["x0", "y0", "x1", "y1", "h0", "dh", "hmap", "cmap", "nmap"], headers=['"common/point_utils.h"'], include_dirs=['cpp'], type_converters=converters.blitz)
    return (cmap, nmap, int(h0), int(dh))


def buildFromVolumeData(src, threshold):
    cmap = zeros(src.shape+(4,), uint8)
    nmap = zeros(src.shape+(4,), int8)
    code = '''
        #line 97 "scene_gen.py"
        using namespace cg;

        int n = Nsrc[0];

        for (walk_3 i(n, n, n); !i.done(); ++i)
        {
          if (src(i.z(), i.y(), i.x()) < threshold)
            continue;
          
          unsigned char col[3] = {128, 128, 128};
          float light = 0.5;
          float dx(0), dy(0), dz(0);
          if (i.x() > 0 && i.x() < n-1)
            dx = 0.5f*( src(i.z(), i.y(), i.x()+1 ) - src(i.z(), i.y(), i.x()-1) );

          if (i.y() > 0 && i.y() < n-1)
            dy = 0.5f*( src(i.z(), i.y()+1, i.x() ) - src(i.z(), i.y()-1, i.x()) );

          if (i.z() > 0 && i.z() < n-1)
            dz = 0.5f*( src(i.z()+1, i.y(), i.x() ) - src(i.z()-1, i.y(), i.x()) );

          point_3f nDir(-dx, -dy, -dz);
          normalize(nDir);
          nDir *= 127.0f;

          cmap(i.z(), i.y(), i.x(), 0) = col[0];
          cmap(i.z(), i.y(), i.x(), 1) = col[1];
          cmap(i.z(), i.y(), i.x(), 2) = col[2];
          cmap(i.z(), i.y(), i.x(), 3) = 255;
          
          nmap(i.z(), i.y(), i.x(), 0) = nDir[0];
          nmap(i.z(), i.y(), i.x(), 1) = nDir[1];
          nmap(i.z(), i.y(), i.x(), 2) = nDir[2];
        }
    '''
    weave.inline(code, ["src", "cmap", "nmap", "threshold"], headers=['"common/grid_walk.h"'], include_dirs=['cpp'], type_converters=converters.blitz)
    return (cmap, nmap)

def fillInternal(cmap, nmap, mark=1):
    src = cmap.copy()
    dst = cmap
    col = array([0, 0, 0, mark], uint8) # 1 means internal
    n = array([0, 0, 0, 0], int8)
    code = '''
        #line 3000 "scene_gen.py"
        for (walk_3 i(Nsrc[0], Nsrc[1], Nsrc[2]); !i.done(); ++i)
        {
            if (src(i.z(), i.y(), i.x(), 3) < 255)
                continue;
            if (i.x() > 0 && src(i.z(), i.y(), i.x()-1, 3) < 255 )
                continue;
            if (i.y() > 0 && src(i.z(), i.y()-1, i.x(), 3) < 255 )
                continue;
            if (i.z() > 0 && src(i.z()-1, i.y(), i.x(), 3) < 255 )
                continue;
            if (i.x() < Nsrc[0]-1 && src(i.z(), i.y(), i.x()+1, 3) < 255 )
                continue;
            if (i.y() < Nsrc[1]-1 && src(i.z(), i.y()+1, i.x(), 3) < 255 )
                continue;
            if (i.z() < Nsrc[2]-1 && src(i.z()+1, i.y(), i.x(), 3) < 255 )
                continue;
            
            dst(i.z(), i.y(), i.x(), blitz::Range::all()) = col;
            nmap(i.z(), i.y(), i.x(), blitz::Range::all()) = n;
        }

    '''
    weave.inline(code, ["src", "dst", "nmap", "col", "n"], headers=['"common/grid_walk.h"'], include_dirs=['cpp'], type_converters=converters.blitz)


def buildHeightmap(bld, hmap, level, pos):
    start = clock()
    step = 8
    for y in xrange(0, hmap.shape[0]/step):
        for x in xrange(0, hmap.shape[1]/step):
            (cmap, nmap, h0, dh) = buildRegion(x*step, y*step, (x+1)*step, (y+1)*step, hmap)
            src = MakeRawSource(point_3i(step, step, dh), cmap, nmap)
            bld.BuildRange(level, point_3i(pos[0] + x*step, pos[1] + y*step, pos[2] + h0), BuildMode.GROW, src)
        print y, bld.nodecount
    dt = clock() - start
    print "land build time: %.2f s" % dt


def addHeightmaps(bld):
    im = Image.open("data/heightmap1024.png")
    hmap = fromstring(im.tostring(), uint16)
    print hmap.shape
    s = im.size
    print s
    hmap.shape = (s[0], s[1], 2)
    hmap = hmap[...,0].astype(float32)
    hmap *= 0.5/256

    buildHeightmap(bld, hmap, 11, (0, 0, 0))
    buildHeightmap(bld, hmap, 11, (768, 768, 256))


if __name__ == '__main__':
    
    bld = DynamicSVO()

    addHeightmaps(bld)

    
    mri = fromfile("data/bonsai.raw", uint8)
    mri.shape = (256, 256, 256)
    mri = rot90(mri, -1)
    print "processing mri data..."
    (cmap, nmap) = buildFromVolumeData(mri, 40)
    print "filling internal space..."
    fillInternal(cmap, nmap)
    print "inserting object into tree..."
   
    treeSrc = MakeRawSource(point_3i(256, 256, 256), cmap, nmap)
    print "tree 1"
    bld.BuildRange(11, point_3i(256, 256, 64), BuildMode.GROW, treeSrc)
    print "tree 2"
    bld.BuildRange(11, point_3i(768, 256, 64), BuildMode.GROW, treeSrc)
    print "tree 3"
    bld.BuildRange(11, point_3i(256, 768, 32), BuildMode.GROW, treeSrc)
    
    print "saving tree..."
    bld.Save("data/scene.vox")
