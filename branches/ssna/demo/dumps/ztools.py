from numpy import *
from scipy import weave

def findEdges(z, thld):
    m = zeros_like(z)                 
    dx = z[:,1:] - z[:,:-1]
    dy = z[1:,:] - z[:-1,:]
    mx = abs(dx) > thld
    my = abs(dy) > thld
    m[:-1][my] = 1
    m[1: ][my] = 1
    m[:,:-1][mx] = 1
    m[:,1: ][mx] = 1
    return m

fov = radians(70.0)
wldVoxelSize = 1.0/2048


def absMin(a, b):
    return choose(abs(a) < abs(b), (b, a))

def calcNormals(z):
    d = 2*tan(fov/2) / z.shape[1]

    #dx = z[:-1,1:] - z[:-1,:-1]
    #dy = z[1:,:-1] - z[:-1,:-1]
    #z = z[:-1, :-1]

    
    dx = z[1:-1,1:] - z[1:-1,:-1]
    dy = z[1:,1:-1] - z[:-1,1:-1]
    dy = absMin(dy[1:], dy[:-1])
    dx = absMin(dx[:,1:], dx[:,:-1])
    z = z[1:-1, 1:-1]
    
    nx = -d * dx * z
    ny = -d * dy * z
    nz = d * d * z * z
    nl = sqrt(nx**2 + ny**2 + nz**2)
    n = dstack((nx, ny, nz))
    n /= nl[...,newaxis]
    return n


def calcLight(n, z, lightPos):
    d = 2*tan(fov/2) / n.shape[1]
    y, x = mgrid[:n.shape[0], :n.shape[1]]
    x -= n.shape[1]/2
    y -= n.shape[0]/2
    v = (x*d, y*d, ones_like(x))
    v = dstack(v)

    v *= z[1:-1, 1:-1][...,newaxis]
    v -= array(lightPos)
    
    vl = sqrt( (v*v).sum(-1) )
    v /= vl[...,newaxis]

    c = (v*n).sum(-1)
    return maximum(c, 0)

    
def lightByZ(z, lightPos = (0, 0, 0)):
    return calcLight(calcNormals(z), z, lightPos)


def makeGauss2d(n):
    from scipy.signal import gaussian
    g = gaussian(n, n / 5.0)
    g = outer(g, g)
    g /= g.sum()
    return g


def filterGauss(z, n):
    from scipy.signal import convolve2d
    return convolve2d(z, makeGauss2d(n))


def calcDiv(z):
    ddx = 2 * z[1:-1, 1:-1] - z[1:-1, 0:-2] - z[1:-1, 2:]
    ddy = 2 * z[1:-1, 1:-1] - z[0:-2, 1:-1] - z[2:, 1:-1]
    res = zeros_like(z)
    res[1:-1, 1:-1] = ddx + ddy
    return res


def divBlur(z, n, c=0.2):
    m = 1 - findEdges(z, 4.0/2048)
    res = z.copy()
    for i in range(n):
        div = calcDiv(res) * m
        res -= div*c
        print i
    return res

def distBlurTest(z, n, passCoef):
    m = 1 - findEdges(z, 4.0/2048)
    ms = zeros_like(m)
    ms[1:] += m[:-1]
    ms[:-1] += m[1:]
    ms[:,1:] += m[:,:-1]
    ms[:,:-1] += m[:,1:]
    ms = maximum(1, ms)

    res = z.copy()

    d = 2*tan(fov/2) / z.shape[1]
    voxSize = 1.0/ (2048.0 * d * z)

    div2 = zeros_like(res)

    for i in range(n):
        div = calcDiv(res) * m
        code = '''
        int sx = Ndiv[1], sy = Ndiv[0];
        for (int y = 1; y < sy-1; ++y)
          for (int x = 1; x < sx-1; ++x)
          {
            int ofs = y*sx+x;
            if (m[ofs] > 0)
            {
              div2[ofs] = div[ofs];
              continue;
            }

            float a = 0;
            a += div[ofs+1];
            a += div[ofs-1];
            a += div[ofs+sx];
            a += div[ofs-sx];
            a /= ms[ofs];
            div2[ofs] = a;
          }

        '''
        #weave.inline(code, ['div', 'div2', 'm', 'ms'])

        m2 = choose(voxSize > (i*passCoef), (0, 1))
        res -= div*m2*0.2
        print i, m2.sum()
    return res

def makeEdgePad(kernelSize, edgeVal):
    y, x = mgrid[:kernelSize, :kernelSize]
    x = 2.0*x/kernelSize - 1.0
    y = 2.0*y/kernelSize - 1.0
    res = x*x + y*y
    res *= edgeVal;
    return res.astype(float32)


def limitedGaussBlur(z, kernelSize, threshold, zrange, res, edgePadCoef):
    z = z.astype(float32)
    kern = makeGauss2d(kernelSize).astype(float32)

    code = '''
      array_2d<float> az(Nz[1], Nz[0], z);
      array_2d<float> ares(Nres[1], Nres[0], res);
      array_2d<float> akern(Nkern[1], Nkern[0], kern);
      //array_2d<float> aedgePad(NedgePad[1], NedgePad[0], edgePad);

      float z_lo = zrange[0];
      float z_hi = zrange[1];
      
      int count = 0;

      for (int y = 0; y < az.height(); ++y)
        for (int x = 0; x < az.width(); ++x)
        {
          int x1 = max(0, x - akern.width()/2);
          int x2 = min(az.width(), x1 + akern.width());
          int y1 = max(0, y - akern.height()/2);
          int y2 = min(az.height(), y1 + akern.height());
          
          float d = az(x, y);
          if (d <= z_lo || d > z_hi)
          {
            ares(x, y) = d;
            continue;
          }
          ++count;

          float acc = 0;
          float c = 0;
          for (int sy = y1; sy < y2; ++sy)
            for (int sx = x1; sx < x2; ++sx)
            {
              float s = az(sx, sy);
              float k = akern(sx - x1, sy - y1);
              if (s > d + threshold)
                s = d + edgePadCoef * threshold;
              else if ( d > s + threshold )
              {
                if (mode == 0)
                  k = 0;
                else 
                if (mode == 1)
                {
                  int kx = sx - x1;
                  int ky = sy - y1;

                  float ssym = az(x2-kx-1, y2-ky-1);
                  if (ssym > s + threshold)
                    k = 0;
                  else
                    d = ssym;
                }
                else if (mode == 2)
                {
                  int kx = sx - x1;
                  int ky = sy - y1;

                  float ssym = az(x2-kx-1, y2-ky-1);
                  if (d > ssym + threshold)
                    k = 0;
                  else
                    s = ssym;
                }

              }
            
              acc += s * k;
              c += k;
            }
          ares(x, y) = acc / c;
        }
      return_val = count;
    '''
    mode = 2
    count = weave.inline(code, ["z", "res", "kern", "threshold", "edgePadCoef", "zrange", "mode"], headers=['"array_2d.h"'], include_dirs=["."])
    print count


def adaptiveGaussBlur(z, threshold):
    res = ones_like(z)
    d = 2*tan(fov/2) / z.shape[1] * 0.33
    prev_z = wldVoxelSize / d
    res[z > prev_z] = z[z > prev_z]

    maxKernSize = 40
    for i in xrange(2, maxKernSize):
        print i
        next_z = wldVoxelSize / (i*d)
        limitedGaussBlur(z, i, threshold, (next_z, prev_z), res)
        prev_z = next_z
    limitedGaussBlur(z, maxKernSize, threshold, (0.0, prev_z), res)

    return res

def multipassGaussBlur(z, threshold):
    buf1 = z.astype(float32)
    buf2 = zeros_like(buf1)
    
    passNum = 30
    kernSize = 5

    d = 2*tan(fov/2) / z.shape[1] * 0.33

    for i in xrange(passNum):
        print i
        z_limit = wldVoxelSize / ((i+1) * d) 
        limitedGaussBlur(buf1, kernSize, threshold, (0.0, z_limit), buf2, 0.3)
        buf1, buf2 = buf2, buf1
    return buf1


def loadDists(fn, sx, sy):
    z = fromfile(fn, float32)
    z.shape = (sy, sx)
    z[z<=0] = 1
    z = flipud(z)

    d = 2*tan(fov/2) / z.shape[1]
    y, x = mgrid[:z.shape[0], :z.shape[1]]
    x -= z.shape[1]/2
    y -= z.shape[0]/2
    vx = x*d
    vy = y*d
    vz = ones_like(x)
    vl = sqrt(vx**2 + vy**2 + vz**2)
    z /= vl
    return z


def loadData(fn):
    fnBase = fn[:fn.find('.')]
    resStr = fnBase[fnBase.rfind('_')+1:]

    ps = resStr.find('x')
    sx = int(resStr[:ps])
    sy = int(resStr[ps+1:])

    z = loadDists(fnBase + '.dist', sx, sy)

    color = fromfile(fnBase + '.color', uint8)
    color.shape = (sy, sx, 4)
    color = flipud( color[..., :3].astype(float32) )
    color /= 255.0;
    return (z, color)

def test1(z, c):
    z1 = distBlurTest(z, 40, 0.3)
    light = lightByZ(z1)
    res = c[1:-1, 1:-1] * light[...,newaxis]

    m = findEdges(z, 4.0/2048)[1:-1,1:-1]
    res[...,0] = choose( m > 0, (res[...,0],1) )
    return res

def test2(z, c, lightPos = (0, 0, 0)):
    #z1 = adaptiveGaussBlur(z, 8.0*wldVoxelSize)
    z1 = multipassGaussBlur(z, 4.0*wldVoxelSize)
    light = lightByZ(z1)
    res = c[1:-1, 1:-1] * light[...,newaxis]
    return res

def saveImage(arr, fn):
    from PIL import Image
    im = Image.fromarray((arr*255).astype(uint8))
    im.save(fn)
    print fn, "saved"


def animTest(z, c):
     z1 = multipassGaussBlur(z, 4.0*wldVoxelSize)
     n = calcNormals(z1)
     for i in xrange(0, 50):
        ang = 2*pi/50*i
        lightPos = ( 0.2*sin(ang), -0.2, (1-cos(ang))*0.2 )
        light = calcLight(n, z1, lightPos)
        res = c[1:-1, 1:-1] * light[...,newaxis]
        saveImage(res, "anim/frame_%02d.bmp" % i)


if __name__ == '__main__':
    from pylab import *
    
    #(z, c) = loadData("dmp_3_1280x968.dist")
    (z, c) = loadData("dmp_7_800x600.dist")
    
    #animTest(z, c)

    res = test2(z, c)
        
    saveImage(res, "out.png")
    #imshow(res, interpolation = 'nearest')
    #show()
    
