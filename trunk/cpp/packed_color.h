#pragma once

typedef ushort Color16;

inline GLOBAL_FUNC Color16 PackColor(rgba c)
{
  Color16 c16 = 0;
  c16 |= (c.r>>3);          // red
  c16 |= (c.g>>2) << 5;     // green
  c16 |= (c.b>>3) << (5+6); // blue
  return c16;
}

inline GLOBAL_FUNC rgba UnpackColor(Color16 c16)
{
  rgba c;
  c.r = (c16 << 3) & 0xf8;
  c.g = (c16 >> (5-2)) & 0xfc;
  c.b = (c16 >> (5+6-3)) & 0xf8;
  c.a = 0;
  return c;
}
