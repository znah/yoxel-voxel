from numpy import *
from PIL import Image
from ore.ore import *
from scipy import weave
from scipy.weave import converters
from time import clock

def buildRegion(x0, y0, x1, y1, hmap, tex):
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

            //point_3f c1(0.1f, 0.5f, 0.1f), c2(0.7f, 0.3f, 0.2f);
            point_3b c(tex(y, x, 0), tex(y, x, 1), tex(y, x, 2));// = 255*(c1 + (h/300.0f) * (c2-c1));
            
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
    weave.inline(code, ["x0", "y0", "x1", "y1", "h0", "dh", "hmap", "cmap", "nmap", "tex"], headers=['"common/point_utils.h"'], include_dirs=['cpp'], type_converters=converters.blitz)
    return (cmap, nmap, int(h0), int(dh))


def buildHeightmap(bld, hmap, tex, level, pos):
    start = clock()
    step = 8
    for y in xrange(0, hmap.shape[0]/step):
        for x in xrange(0, hmap.shape[1]/step):
            (cmap, nmap, h0, dh) = buildRegion(x*step, y*step, (x+1)*step, (y+1)*step, hmap, tex)
            src = MakeRawSource(point_3i(step, step, dh), cmap, nmap)
            bld.BuildRange(level, point_3i(pos[0] + x*step, pos[1] + y*step, pos[2] + h0), BuildMode.GROW, src)
        print y, bld.nodecount
    dt = clock() - start
    print "land build time: %.2f s" % dt


def addHeightmaps(bld):
    hmap = asarray(Image.open("data/gcanyon_height.png"))
    tex = asarray(Image.open("data/gcanyon_color_4k2k.png"))
    
    n = 2048
    hmap = hmap[:n, :n]
    tex = tex[:n, :n]

    #import pylab
    #pylab.imshow(hmap)
    #pylab.show()
    hmap = hmap.astype(float32)
    hmap *= 2.0
    from scipy import signal
    g = signal.gaussian(5, 1.0)
    g = outer(g, g)
    g /= g.sum()
    hmap2 = signal.convolve2d(hmap, g, "same", "symm")


    buildHeightmap(bld, hmap2, tex, 11, (0, 0, 0))
    #buildHeightmap(bld, hmap, 11, (768, 768, 256))


def addTrees(bld):
    print "loading mri data..."
    mri = fromfile("data/bonsai.raw", uint8)
    mri.shape = (256, 256, 256)
    mri = rot90(mri, -1).copy()
    
    print "creating IsoSource..."
    mriSrc = MakeIsoSource(p3i(mri.shape), mri)

    print "adding tree 1"
    mriSrc.SetIsoLevel(40)
    bld.BuildRange(11, point_3i(256, 256, 64), BuildMode.GROW, mriSrc)

    print "adding tree 2"
    mriSrc.SetIsoLevel(60)
    bld.BuildRange(11, point_3i(768, 256, 64), BuildMode.GROW, mriSrc)
    
    print "adding tree 3"
    mriSrc.SetIsoLevel(80)
    bld.BuildRange(11, point_3i(256, 768, 64), BuildMode.GROW, mriSrc)


if __name__ == '__main__':
    
    bld = DynamicSVO()

    addHeightmaps(bld)
    #addTrees(bld)
    
    print "saving tree..."
    bld.Save("data/scene.vox")
