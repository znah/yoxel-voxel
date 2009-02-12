#pragma once

struct matrix_4f
{
  float data[4][4];

  GLOBAL_FUNC void setcol(int j, float v0, float v1, float v2, float v3)
  {
    data[0][j] = v0;
    data[1][j] = v1;
    data[2][j] = v2;
    data[3][j] = v3;
  }

  GLOBAL_FUNC void setcol(int j, const point_3f & p, float w)
  {
    setcol(j, p.x, p.y, p.z, w);
  }
  
  GLOBAL_FUNC const float & operator()(int i, int j) const { return data[i][j]; }
  GLOBAL_FUNC float & operator()(int i, int j) { return data[i][j]; }
};

inline GLOBAL_FUNC point_3f operator* (const matrix_4f & mtx, const point_3f & v)
{
  float a[4] = {v.x, v.y, v.z, 1.0f};
  float res[4] = {0, 0, 0, 0};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      res[i] += mtx(i, j) * a[j];
  return point_3f(res[0], res[1], res[2]);
}
