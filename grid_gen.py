from numpy import *
from scipy import weave
from scipy.weave import converters

class GridBuilder:
    def __init__(self, size):
        self.a = zeros((size, size, size, 4), uint8)
        self.setPattern((127, 127, 127, 255), (127, 255, 127, 255), 16)
    
    def setPattern(self, col1, col2, step):
        self.col1 = array(col1, uint8)
        self.col2 = array(col2, uint8)
        self.step = max(step, 1)

    def setPoint(self, p, col):
        self.a[ p[2], p[1], p[0] ] = col

    def addFrame(self):
        a = self.a
        a[ [0,-1,0,-1], [0,0,-1,-1], : ] = (255, 0, 0, 255)
        a[ [0,-1,0,-1], :, [0,0,-1,-1] ] = (0, 255, 0, 255)
        a[ :, [0,-1,0,-1], [0,0,-1,-1] ] = (0, 0, 255, 255)
    
    def addSphere(self, pos, r):
        (pos, rv) = ( array(pos), array([r]*3) )
        p1 = maximum(pos-rv, [0]*3)
        p2 = minimum(pos+rv, self.a.shape[:3])
        code = '''
            #line 1000 "scene_gen.py"
            using namespace cg;

            point_3f lightDir(1, 1, 1);
            normalize(lightDir);

            int r2 = r*r;
            for (walk_3 p( p2(0)-p1(0)+1, p2(1)-p1(1)+1, p2(2)-p1(2)+1 ); !p.done(); ++p)
            {
                int x=p1(0)+p.x(), y=p1(1)+p.y(), z=p1(2)+p.z();
                int dz = z-pos(2);
                int dy = y-pos(1);
                int dx = x-pos(0);
                int d2 = dx*dx + dy*dy + dz*dz;

                point_3f n(dx, dy, dz);
                normalize(n);
                float light = max(n*lightDir, 0.0f);
                light = 0.7f*light + 0.3f;

                if (d2 <= r2)
                {
                    int s = x/step + y/step + z/step;
                    if (s%2 == 0)
                        a(z, y, x, blitz::Range::all()) = col1;
                    else
                        a(z, y, x, blitz::Range::all()) = col2;
                    
                    a(z, y, x, 0) *= light;
                    a(z, y, x, 1) *= light;
                    a(z, y, x, 2) *= light;
                }
            }
        '''
        (a, col1, col2, step) = (self.a, self.col1, self.col2, self.step)
        weave.inline(code, ["a", "col1", "col2", "p1", "p2", "step", "pos", "r"], headers=['"common/grid_walk.h"'], include_dirs=['cpp'], type_converters=converters.blitz)
        
    def fillInternal(self, col):
        col = array(col, uint8)
        a = self.a
        code = '''
            #line 3000 "scene_gen.py"
            for (walk_3 i(Na[0], Na[1], Na[2]); !i.done(); ++i)
            {
                if (a(i.z(), i.y(), i.x(), 3) < 255)
                    continue;
                if (i.x() > 0 && a(i.z(), i.y(), i.x()-1, 3) < 255 )
                    continue;
                if (i.y() > 0 && a(i.z(), i.y()-1, i.x(), 3) < 255 )
                    continue;
                if (i.z() > 0 && a(i.z()-1, i.y(), i.x(), 3) < 255 )
                    continue;
                if (i.x() < Na[0]-1 && a(i.z(), i.y(), i.x()+1, 3) < 255 )
                    continue;
                if (i.y() < Na[1]-1 && a(i.z(), i.y()+1, i.x(), 3) < 255 )
                    continue;
                if (i.z() < Na[2]-1 && a(i.z()+1, i.y(), i.x(), 3) < 255 )
                    continue;
                
                a(i.z(), i.y(), i.x(), blitz::Range::all()) = col;
            }

        '''
        weave.inline(code, ["a", "col"], headers=['"common/grid_walk.h"'], include_dirs=['cpp'], type_converters=converters.blitz)

    def buildFromVolumeData(self, src, threshold, lightDir=(1, 1, 1)):
        dst = zeros(src.shape+(4,), uint8)
        code = '''
            #line 97 "scene_gen.py"
            using namespace cg;

            point_3f lDir(lightDir[0], lightDir[1], lightDir[2]);
            normalize(lDir);
            int n = Nsrc[0];

            for (walk_3 i(n, n, n); !i.done(); ++i)
            {
              if (src(i.z(), i.y(), i.x()) < threshold)
                continue;
              
              unsigned char col[3] = {255, 255, 255};
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

              light += 0.5f * max(nDir * lDir, 0.0f);

              dst(i.z(), i.y(), i.x(), 0) = col[0]*light;
              dst(i.z(), i.y(), i.x(), 1) = col[1]*light;
              dst(i.z(), i.y(), i.x(), 2) = col[2]*light;
              dst(i.z(), i.y(), i.x(), 3) = 255;
            }
        '''
        weave.inline(code, ["src", "dst", "threshold", "lightDir"], headers=['"common/grid_walk.h"'], include_dirs=['cpp'], type_converters=converters.blitz)
        self.a = dst

        
    def getResult(self):
        return self.a


def renderGrid(a, res, viewDir, first=False):
    n = a.shape[0]
    img = zeros((res, res, 3), uint8)
    code = '''
        #line 2000 "scene_gen.py"
        for (OrthoRayGen rg(viewDir, res); !rg.done(); rg.next())
        {
          grid3_tracer tr(n, rg.p0(), rg.dir());
          for (; !tr.done(); tr.next())
          {
            cg::point_3i p = tr.p();
            if (a(p.z, p.y, p.x, 3) > 0)
            {
              img(rg.y(), rg.x(), 0) = a(p.z, p.y, p.x, 0);
              img(rg.y(), rg.x(), 1) = a(p.z, p.y, p.x, 1);
              img(rg.y(), rg.x(), 2) = a(p.z, p.y, p.x, 2);
              break;
            }
          }
        }
    '''
    weave.inline(code, ["n", "a", "res", "viewDir", "img"], headers=['"grid_trace.h"', '"ray_gen.h"'], include_dirs=['cpp'], type_converters=converters.blitz, force=first)
    return img

def genBallsScene(n, seed=None, noise=True):
    from numpy.random import rand
    random.seed(seed)
    gen = GridBuilder(n)
    
    for i in xrange(40):
        c1 = (rand(4)*256).astype(uint8)
        c2 = (rand(4)*256).astype(uint8)
        c1[3] = 255
        c2[3] = 255
        gen.setPattern(c1, c2, n/16) 
        pos = rand(3)*n
        r = rand()*n/8
        gen.addSphere(pos, r)

    c = (255, 255, 255, 255)
    gen.setPattern(c, c, n/16) 
    gen.addSphere((0, 0, 0), n/16)

    print "spheres added"
    
    if noise:
        for i in xrange(2000):
            p = rand(3)*n
            gen.setPoint(p, (128, 128, 128, 255))
        print "noise added"

    gen.fillInternal((64, 0, 0, 255))
    print "internals filled"
    #gen.addFrame()
    return gen.getResult()

def genMRIScene(src, threshold, lightDir=(1, 1, 1)):
    gen = GridBuilder(1)
    gen.buildFromVolumeData(src, threshold, lightDir)
    print "voxels created"

    gen.fillInternal((64, 0, 0, 255))
    print "internals filled"

    gen.addFrame()
    return gen.getResult()

   
if __name__ == '__main__':
    #a = genBallsScene(256, 141593)

    src = fromfile("data/bonsai.raw", dtype=uint8)
    src.shape = (256, 256, 256)
    gen = GridBuilder(256)
    gen.buildFromVolumeData(src, 40)


    #from render import renderTurn
    #renderTurn(a, renderGrid)
