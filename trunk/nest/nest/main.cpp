#include "stdafx.h"
#include "ntree.h"


/*void MakeShpere(int size, array_3d<uint8> & dst)
{
  point_3i sz(size, size, size);
  dst.resize(sz);
  point_3f c = 0.5f * (sz - point_3i(1, 1, 1));
  const float dl = 2.0f * sqrt(3.0f);
  const float invDl = 1.0f / dl;
  const float l0 = 0.5f * size - dl;
  for (walk_3 i(sz); !i.done(); ++i)
  {
    point_3f d = i.p - c;
    float l = cg::norm(d);
    float v = 1.0f - (l - l0) * invDl;
    v = cg::bound(v, 0.0f, 1.0f);
    dst[i.p] = (uint8)(v*255.0f);
  }
}

void save(array_3d_ref<uint8> & a, const char * fn)
{
  std::ofstream file(fn, std::ios::binary);
  write(file, a.extent());
  file.write((char *)a.data(), a.size() * sizeof(uint8));
}*/

