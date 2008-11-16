#pragma once

struct point_3f : public float3
{
  GLOBAL_FUNC point_3f(float x_, float y_, float z_) { x = x_; y = y_; z = z_; }

  GLOBAL_FUNC float & operator[](int i) {return *(&x+i); }  // works well on CUDA only if 'i' is knows at compile time
  GLOBAL_FUNC float operator[](int i) const {return *(&x+i); }  // works well on CUDA only if 'i' is knows at compile time
  GLOBAL_FUNC point_3f(const float3 & p) { x = p.x; y = p.y; z = p.z; }
  GLOBAL_FUNC point_3f() { x = 0; y = 0; z = 0; }

}; 

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
