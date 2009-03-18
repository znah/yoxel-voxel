from numpy import *
from scipy import weave
from scipy.weave import converters


src = fromfile("data/bonsai.raw", uint8)
src.shape = (256, 256, 256)
dst = zeros_like(src)

level = 50;

code = '''
    #line 13 "scene_gen.py"
 
    int nz = src.extent(0);
    int ny = src.extent(1);
    int nx = src.extent(2);
    for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x)
    {
      int v = src(z, y, x);
      int lo = v, hi = v;
      for (int dz = -1; dz < 2; ++dz)
      for (int dy = -1; dy < 2; ++dy)
      for (int dx = -1; dx < 2; ++dx)
      {
        //if (abs(dx) + abs(dy) + abs(dz) == 3)
        //  continue;
        int x1 = x + dx, y1 = y + dy, z1 = z + dz;
        if (x1 < 0 || x1 >= nx || y1 < 0 || y1 >= ny || z1 < 0 || z1 >= nz)
          continue;
        int v1 = src(z1, y1, x1);
        lo = min(lo, v1);
        hi = max(hi, v1);
      }
      if (v >= level && lo >= level)
        v = 255;
      else if (v < level && hi < level)
        v = 0;

      dst(z, y, x) = v;
    }
'''
weave.inline(code, ["src", "dst", "level"], type_converters=converters.blitz)


dst.tofile("data/bonsai32.raw")