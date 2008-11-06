#pragma once

typedef ushort Color16;

inline GLOBAL_FUNC Color16 PackColor(Color32 c)
{
  Color16 c16 = 0;
  c16 |= (c[0]>>3);          // red
  c16 |= (c[1]>>2) << 5;     // green
  c16 |= (c[2]>>3) << (5+6); // blue
  return c16;
}

inline GLOBAL_FUNC Color32 UnpackColor(Color16 c16)
{
  Color32 c;
  c[0] = (c16 << 3) & 0xf8;
  c[1] = (c16 >> (5-2)) & 0xfc;
  c[2] = (c16 >> (5+6-3)) & 0xf8;
  c[3] = 0;
  return c;
}

#ifdef __CUDACC__ 

inline GLOBAL_FUNC uchar4 UnpackColorCU(Color16 c16)
{
  uchar4 c;
  c.x = (c16 << 3) & 0xf8;
  c.y = (c16 >> (5-2)) & 0xfc;
  c.z = (c16 >> (5+6-3)) & 0xf8;
  c.w = 0;
  return c;
}

#endif
