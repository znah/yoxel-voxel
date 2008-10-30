#pragma once

typedef ushort Color16;

inline GLOBAL_FUNC Color16 PackColor(uchar4 c)
{
  Color16 c16 = 0;
  c16 |= (c.x>>3);          // red
  c16 |= (c.y>>2) << 5;     // green
  c16 |= (c.z>>3) << (5+6); // blue
  return c16;
}

inline GLOBAL_FUNC uchar4 UnpackColor(Color16 c16)
{
  uchar4 c;
  c.x = (c16 << 3) & 0xf8;
  c.y = (c16 >> (5-2)) & 0xfc;
  c.z = (c16 >> (5+6-3)) & 0xf8;
  c.w = 0;
  return c;
}
