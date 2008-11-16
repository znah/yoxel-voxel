#pragma once

struct cprf
{
  float crs, pitch, roll;

  cprf() : crs(0), pitch(0), roll(0) {}
  cprf(float c, float p, float r) : crs(c), pitch(p), roll(r) {}
};
