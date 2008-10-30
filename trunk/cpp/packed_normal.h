#pragma once

// octant - 3bit, qx - 7 bit, qy - 6 bit
typedef ushort Normal16;

inline GLOBAL_FUNC Normal16 PackNormal(float x, float y, float z)
{
  int octant = 0;
  if (x < 0) { octant |= 1; x = -x; }
  if (y < 0) { octant |= 2; y = -y; }
  if (z < 0) { octant |= 4; z = -z; }

  float t = 1.0f / (x + y + z);
  float px = t*x;
  float py = t*y;

  int qx = (int)(127.0f*px);
  int qy = (int)(63.0f*py);

  Normal16 res = 0;
  res |= octant;
  res |= qx << 3;
  res |= qy << (3+7);
  return res;
}

inline GLOBAL_FUNC void UnpackNormal(Normal16 packed, float & x, float & y, float & z)
{
  int octant = packed & 7;
  int qx = (packed >> 3) & 127;
  int qy = (packed >> (3+7)) & 63;
  
  x = qx / 127.0f;
  y = qy / 63.0f;
  z = 1.0f - x - y;
  float invLen = 1.0f / sqrtf(x*x + y*y + z*z);
  x *= invLen;
  y *= invLen;
  z *= invLen;

  if (octant & 1) x = -x;
  if (octant & 2) y = -y;
  if (octant & 4) z = -z;
}
