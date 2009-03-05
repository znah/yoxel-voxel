#pragma once

inline GLOBAL_FUNC float maxCoord(const point_3f & p) { return max(p.x, max(p.y, p.z)); }
inline GLOBAL_FUNC float minCoord(const point_3f & p) { return min(p.x, min(p.y, p.z)); }



inline GLOBAL_FUNC void AdjustDir(point_3f & dir)
{
  const float eps = 1e-8f;
  for (int i = 0; i < 3; ++i)
    if (fabs(dir[i]) < eps)
      dir[i] = (dir[i] < 0) ? -eps : eps;
}

inline GLOBAL_FUNC bool SetupTrace(const point_3f & p, const point_3f & dir, point_3f & t1, point_3f & t2, uint & dirFlags)
{
  t1 = (point_3f(0.0f, 0.0f, 0.0f) - p) / dir;
  t2 = (point_3f(1.0f, 1.0f, 1.0f) - p) / dir;
  dirFlags = 0;
  for (int i = 0; i < 3; ++i)
    if (dir[i] < 0)
    {
      dirFlags |= 1<<i;
      swap(t1[i], t2[i]);
    }

  float tenter = maxCoord(t1);
  float texit  = minCoord(t2);
  return tenter < texit && texit > 0;
}



inline GLOBAL_FUNC int FindFirstChild(point_3f & t1, point_3f & t2)
{
  int childId = 0;
  point_3f tm = 0.5f * (t1 + t2);
  float tEnter = maxCoord(t1);
  for (int i = 0; i < 3; ++i)
  {
    if (tm[i] > tEnter)
      t2[i] = tm[i];
    else
    {
      t1[i] = tm[i];
      childId |= 1<<i;
    }
  }
  return childId;
}

template<int ExitPlane>
inline GLOBAL_FUNC bool GoNextTempl(int & childId, point_3f & t1, point_3f & t2)
{
  int mask = 1<<ExitPlane;
  if ((childId & mask) != 0)
    return false;

  childId ^= mask;

  float dt = t2[ExitPlane] - t1[ExitPlane];
  t1[ExitPlane] = t2[ExitPlane];
  t2[ExitPlane] += dt;
  return true;
}

inline GLOBAL_FUNC  bool GoNext(int & childId, point_3f & t1, point_3f & t2)
{
  // argmin
  if (t2.x > t2.y)
    return (t2.y < t2.z) ? GoNextTempl<1>(childId, t1, t2) : GoNextTempl<2>(childId, t1, t2);
  else
    return (t2.x < t2.z) ? GoNextTempl<0>(childId, t1, t2) : GoNextTempl<2>(childId, t1, t2);
}

inline GLOBAL_FUNC  void GoUp(int childId, point_3f & t1, point_3f & t2)
{
  for (int i = 0; i < 3; ++i)
  {
    int mask = 1<<i;
    float dt = t2[i] - t1[i];
    ((childId & mask) == 0) ? t2[i] += dt : t1[i] -= dt;
  }
}

