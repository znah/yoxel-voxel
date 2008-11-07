#pragma once

typedef ushort Color16;

inline GLOBAL_FUNC COLOR16 PackColor(uchar r, uchar g, uchar b)
{
  Color16 c16 = 0;
  c16 |= (r>>3);          // red
  c16 |= (g>>2) << 5;     // green
  c16 |= (b>>3) << (5+6); // blue
  return c16;
}

inline GLOBAL_FUNC void UnpackColor(Color16 c16, uchar &r, uchar &g, uchar &b)
{
  r = (c16 << 3) & 0xf8;
  g = (c16 >> (5-2)) & 0xfc;
  b = (c16 >> (5+6-3)) & 0xf8;
}


#ifdef USE_CG
inline GLOBAL_FUNC Color16 PackColor(Color32 c)
{
  return PackColor(c[0], c[1], c[2]);
}

inline GLOBAL_FUNC Color32 UnpackColor(Color16 c16)
{
  Color32 c;
  UnpackColor(c16, c[0], c[1], c[2]);
  return c;
}
#endif


#ifdef TARGET_CUDA
inline GLOBAL_FUNC uchar4 UnpackColorCU(Color16 c16)
{
  uchar4 c;
  UnpackColor(c16, c.x, c.y, c.z);
  c.w = 0;
  return c;
}
#endif
