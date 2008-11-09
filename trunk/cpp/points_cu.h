#pragma once

struct point_3f : public float3
{
  GLOBAL_FUNC float & operator[](int i) {return *(&x+i); }
  GLOBAL_FUNC point_3f(const float3 & p) { x = p.x; y = p.y; z = p.z; }
  GLOBAL_FUNC point_3f() { x = 0; y = 0; z = 0; }

}; 

inline GLOBAL_FUNC float max(const float3 & p) { return fmaxf(p.x, fmaxf(p.y, p.z)); }
inline GLOBAL_FUNC float min(const float3 & p) { return fminf(p.x, fminf(p.y, p.z)); }

inline GLOBAL_FUNC int argmin(const float3 & p) 
{
  if (p.x > p.y)
    return (p.y < p.z) ? 1 : 2;
  else
    return (p.x < p.z) ? 0 : 2;
}

inline GLOBAL_FUNC int argmax(const float3 & p) 
{
  if (p.x < p.y)
    return (p.y > p.z) ? 1 : 2;
  else
    return (p.x > p.z) ? 0 : 2;
}
